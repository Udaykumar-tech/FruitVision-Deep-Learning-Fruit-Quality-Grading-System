import io
import time
import json
import base64
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from models import EffNetSwinHybrid, DBAViNet, MobileNetV4Wrapper, NUM_CLASSES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_SIZE = 256
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

BASE_DIR = Path(__file__).parent.parent
CKPT_DIR = BASE_DIR / "checkpoints"
SPLIT_DIR = BASE_DIR / "splits"

with open(SPLIT_DIR / "label_map.json", "r") as f:
    label_map = json.load(f)
CLASS_NAMES = [label_map["idx_to_label"][str(i)] for i in range(NUM_CLASSES)]

MODELS_INFO = {
    "EffNet+SwinT": {
        "ckpt": "effnet_swin_pathw_best.pt",
        "class_factory": EffNetSwinHybrid,
        "instance": None,
        "cam_target": None,
        "params_M": 48.8,
        "test_acc": 0.9685,
        "focus_lift": 1.46,
        "use_case": "Cloud / high-accuracy server",
    },
    "DBA-ViNet": {
        "ckpt": "dba_vinet_pathw_best.pt",
        "class_factory": DBAViNet,
        "instance": None,
        "cam_target": None,
        "params_M": 12.0,
        "test_acc": 0.8896,
        "focus_lift": 2.82,
        "use_case": "Audit / explainability",
    },
    "MobileNetV4": {
        "ckpt": "mobilenetv4_pathw_best.pt",
        "class_factory": MobileNetV4Wrapper,
        "instance": None,
        "cam_target": None,
        "params_M": 37.0,
        "test_acc": 0.9498,
        "focus_lift": 4.62,
        "use_case": "Mobile / edge deployment",
    },
}


def load_all_models():
    print(f"[inference] Loading models on device: {DEVICE}")
    for name, info in MODELS_INFO.items():
        ckpt_path = CKPT_DIR / info["ckpt"]
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
        print(f"[inference]   Loading {name} from {ckpt_path.name}...")
        t0 = time.time()
        model = info["class_factory"]()
        state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(state["model_state"])
        model = model.to(DEVICE).eval()
        info["instance"] = model
        if name == "EffNet+SwinT":
            info["cam_target"] = [model.effnet.blocks[-1]]
        elif name == "DBA-ViNet":
            info["cam_target"] = [model.global_backbone.blocks[-1]]
        elif name == "MobileNetV4":
            if hasattr(model.backbone, 'blocks') and len(model.backbone.blocks) > 0:
                info["cam_target"] = [model.backbone.blocks[-1]]
            elif hasattr(model.backbone, 'conv_head'):
                info["cam_target"] = [model.backbone.conv_head]
            else:
                info["cam_target"] = [list(model.backbone.children())[-1]]
        print(f"[inference]   {name} ready in {time.time()-t0:.1f}s "
              f"(val_acc={state['val_acc']:.4f})")

    if DEVICE.type == "cuda":
        print(f"[inference] VRAM allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print(f"[inference] All 3 models loaded successfully on {DEVICE}")


def preprocess_image(image_bytes: bytes):
    pil_img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img_orig = np.array(pil_img)
    img_256 = cv2.resize(img_orig, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)
    img_f = img_256.astype(np.float32) / 255.0
    img_norm = (img_f - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
    return tensor, img_256, img_f


def encode_image_to_base64(rgb_uint8: np.ndarray) -> str:
    bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
    success, buf = cv2.imencode(".png", bgr)
    if not success:
        return ""
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def predict_with_model(model, cam_target_layers, input_tensor, img_f):
    t_start = time.time()
    with torch.no_grad():
        if DEVICE.type == "cuda":
            with torch.amp.autocast('cuda'):
                logits = model(input_tensor)
        else:
            logits = model(input_tensor)
        probs = F.softmax(logits.float(), dim=-1)[0].cpu().numpy()
    inference_ms = (time.time() - t_start) * 1000

    top_indices = probs.argsort()[::-1][:2]
    top1_idx = int(top_indices[0])
    top1_conf = float(probs[top1_idx])
    top2_idx = int(top_indices[1])
    top2_conf = float(probs[top2_idx])

    t_cam_start = time.time()
    targets = [ClassifierOutputTarget(top1_idx)]
    with GradCAM(model=model, target_layers=cam_target_layers) as cam:
        gcam = cam(input_tensor=input_tensor, targets=targets)[0]
    overlay = show_cam_on_image(img_f, gcam, use_rgb=True)
    cam_ms = (time.time() - t_cam_start) * 1000

    gradcam_b64 = encode_image_to_base64(overlay)

    return {
        "top1_idx": top1_idx,
        "top1_label": CLASS_NAMES[top1_idx],
        "top1_conf": top1_conf,
        "top2_idx": top2_idx,
        "top2_label": CLASS_NAMES[top2_idx],
        "top2_conf": top2_conf,
        "inference_ms": inference_ms,
        "cam_ms": cam_ms,
        "gradcam_image": gradcam_b64,
    }


def format_prediction_result(raw: dict) -> dict:
    """Apply business logic for top-1/top-2 display."""
    LOW_CONFIDENCE_THRESHOLD = 0.35
    SINGLE_PREDICTION_THRESHOLD = 0.75

    primary = {
        "label": raw["top1_label"].replace("_", " — "),
        "raw_label": raw["top1_label"],
        "confidence": raw["top1_conf"],
        "confidence_pct": raw["top1_conf"] * 100,
        "is_low_confidence": raw["top1_conf"] < LOW_CONFIDENCE_THRESHOLD,
    }
    if primary["is_low_confidence"]:
        primary["label"] = "Object Not Detected"
        primary["raw_label"] = None

    show_secondary = (
        raw["top1_conf"] < SINGLE_PREDICTION_THRESHOLD
        and not primary["is_low_confidence"]
    )

    secondary = None
    if show_secondary:
        if raw["top2_conf"] < LOW_CONFIDENCE_THRESHOLD:
            secondary = {
                "label": "Object Not Detected",
                "raw_label": None,
                "confidence": raw["top2_conf"],
                "confidence_pct": raw["top2_conf"] * 100,
                "is_low_confidence": True,
            }
        else:
            secondary = {
                "label": raw["top2_label"].replace("_", " — "),
                "raw_label": raw["top2_label"],
                "confidence": raw["top2_conf"],
                "confidence_pct": raw["top2_conf"] * 100,
                "is_low_confidence": False,
            }

    return {
        "primary": primary,
        "secondary": secondary,
        "inference_ms": round(raw["inference_ms"], 1),
        "cam_ms": round(raw["cam_ms"], 1),
        "gradcam_image": raw["gradcam_image"],
    }


def predict_all_models(image_bytes: bytes) -> dict:
    request_start = time.time()
    input_tensor, img_256, img_f = preprocess_image(image_bytes)
    preprocess_ms = (time.time() - request_start) * 1000

    results = {}
    for name, info in MODELS_INFO.items():
        raw = predict_with_model(info["instance"], info["cam_target"], input_tensor, img_f)
        formatted = format_prediction_result(raw)
        formatted["model_meta"] = {
            "name": name,
            "params_M": info["params_M"],
            "test_acc": info["test_acc"],
            "focus_lift": info["focus_lift"],
            "use_case": info["use_case"],
        }
        results[name] = formatted

    total_request_ms = (time.time() - request_start) * 1000

    return {
        "results": results,
        "total_request_ms": round(total_request_ms, 1),
        "preprocess_ms": round(preprocess_ms, 1),
        "device": str(DEVICE),
        "image_size": TARGET_SIZE,
    }