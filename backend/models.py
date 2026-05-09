import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

NUM_CLASSES = 15


class EffNetSwinHybrid(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, pretrained=False):
        super().__init__()
        self.effnet = timm.create_model(
            'tf_efficientnetv2_s.in1k', pretrained=pretrained,
            num_classes=0, global_pool=''
        )
        self.swin = timm.create_model(
            'swinv2_tiny_window8_256.ms_in1k', pretrained=pretrained,
            num_classes=0, global_pool='avg'
        )
        self.eff_dim = self.effnet.num_features
        self.swin_dim = self.swin.num_features
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        d = self.eff_dim + self.swin_dim
        self.norm = nn.LayerNorm(d)
        self.dropout1 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(d, 512)
        self.act = nn.GELU()
        self.dropout2 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        eff = self.flatten(self.global_pool(self.effnet.forward_features(x)))
        sw = self.swin(x)
        h = self.norm(torch.cat([eff, sw], -1))
        h = self.fc1(self.dropout1(h))
        return self.fc2(self.dropout2(self.act(h)))


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(dim, dim)
        self.kv_proj = nn.Linear(dim, dim * 2)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, q, kv):
        B, Nq, C = q.shape
        _, Nk, _ = kv.shape
        q_ = self.q_proj(q).view(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        kv_ = self.kv_proj(kv).view(B, Nk, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv_[0], kv_[1]
        attn = (q_ @ k.transpose(-2, -1)) * self.scale
        return self.out_proj((attn.softmax(-1) @ v).transpose(1, 2).reshape(B, Nq, C))


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        h = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, h), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h, dim), nn.Dropout(dropout)
        )

    def forward(self, x):
        a, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)
        x = x + a
        return x + self.mlp(self.norm2(x))


class DBAViNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, pretrained=False,
                 local_dim=192, num_local_blocks=4, num_heads=4):
        super().__init__()
        self.global_backbone = timm.create_model(
            'tf_efficientnetv2_b1.in1k', pretrained=pretrained,
            features_only=True, out_indices=[3, 4]
        )
        ch = self.global_backbone.feature_info.channels()
        self.mid_channels = ch[0]
        self.high_channels = ch[1]
        self.attn_gate = nn.Sequential(
            nn.Conv2d(self.high_channels, self.high_channels // 4, 1),
            nn.GELU(),
            nn.Conv2d(self.high_channels // 4, 1, 1),
            nn.Sigmoid()
        )
        self.local_proj = nn.Conv2d(self.mid_channels, local_dim, 1)
        self.local_pos_embed = nn.Parameter(torch.zeros(1, 16 * 16, local_dim))
        self.local_blocks = nn.ModuleList([
            TransformerBlock(local_dim, num_heads, 2.0)
            for _ in range(num_local_blocks)
        ])
        self.local_norm = nn.LayerNorm(local_dim)
        self.global_proj = nn.Linear(self.high_channels, local_dim)
        self.cross_attn = CrossAttention(local_dim, num_heads)
        self.fusion_norm = nn.LayerNorm(local_dim)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        d = self.high_channels + local_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(d), nn.Dropout(0.2),
            nn.Linear(d, 256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        feats = self.global_backbone(x)
        mid_feat, high_feat = feats[0], feats[1]
        am = self.attn_gate(high_feat)
        if mid_feat.shape[-1] != 16:
            mid_feat = F.interpolate(mid_feat, size=16, mode='bilinear', align_corners=False)
        if am.shape[-1] != 16:
            am_local = F.interpolate(am, size=16, mode='bilinear', align_corners=False)
        else:
            am_local = am
        lf = self.local_proj(mid_feat) * am_local
        B, C, H, W = lf.shape
        tok = lf.flatten(2).transpose(1, 2) + self.local_pos_embed[:, :H * W]
        for blk in self.local_blocks:
            tok = blk(tok)
        tok = self.local_norm(tok)
        gp = self.flatten(self.global_pool(high_feat))
        gt = self.global_proj(gp).unsqueeze(1)
        la = self.fusion_norm(self.cross_attn(tok, gt))
        return self.classifier(torch.cat([gp, la.mean(1)], -1))


class MobileNetV4Wrapper(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, pretrained=False):
        super().__init__()
        candidates = [
            'mobilenetv4_hybrid_large.ix_e600_r384_in1k',
            'mobilenetv4_hybrid_large.e600_r384_in1k',
            'mobilenetv4_hybrid_medium.ix_e550_r256_in1k',
            'mobilenetv4_hybrid_medium.e500_r224_in1k',
            'mobilenetv4_conv_large.e600_r384_in1k',
            'mobilenetv4_conv_large.e500_r256_in1k',
        ]
        self.backbone = None
        for name in candidates:
            try:
                self.backbone = timm.create_model(
                    name, pretrained=pretrained, num_classes=0, global_pool=''
                )
                self.variant_name = name
                break
            except Exception:
                continue
        if self.backbone is None:
            raise RuntimeError("No MobileNetV4 variant could be loaded")
        self.feat_dim = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.LayerNorm(self.feat_dim), nn.Dropout(0.2),
            nn.Linear(self.feat_dim, 512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.backbone.forward_features(x))