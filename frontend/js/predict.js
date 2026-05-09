const MODEL_COLORS = {
  'EffNet+SwinT': { primary: '#60a5fa', dim: 'rgba(96, 165, 250, 0.12)' },
  'DBA-ViNet':    { primary: '#fb923c', dim: 'rgba(251, 146, 60, 0.12)' },
  'MobileNetV4':  { primary: '#4ade80', dim: 'rgba(74, 222, 128, 0.12)' },
};

const MODEL_DESCRIPTIONS = {
  'EffNet+SwinT': 'Parallel ConvNet + Transformer fusion · 49M params · Highest accuracy',
  'DBA-ViNet':    'Dual-branch attention-guided · 12M params · Best size/focus tradeoff',
  'MobileNetV4':  'Mobile-optimized · 37M params · Best overall (Pareto-optimal)',
};

let webcamCtrl = null;
let currentMode = 'upload';
let lastCapturedBlob = null;
let lastCapturedDataUrl = null;

document.addEventListener('DOMContentLoaded', async () => {
  await initHealthCheck();
  initModeToggle();
  initUpload();
  initWebcam();
  initPredictButton();
});

async function initHealthCheck() {
  const indicator = document.getElementById('server-status');
  const indicatorText = document.getElementById('server-status-text');
  try {
    const health = await checkHealth();
    if (health.status === 'ok') {
      indicator.classList.add('bg-green-500');
      indicatorText.textContent = `Server ready · ${health.models_loaded}/3 models on ${health.device}`;
    } else {
      indicator.classList.add('bg-yellow-500');
      indicatorText.textContent = `Partial: ${health.models_loaded}/3 models loaded`;
    }
  } catch (err) {
    indicator.classList.add('bg-red-500');
    indicatorText.textContent = 'Server not responding';
  }
}

function initModeToggle() {
  const uploadBtn  = document.getElementById('mode-upload');
  const webcamBtn  = document.getElementById('mode-webcam');
  const uploadView = document.getElementById('upload-view');
  const webcamView = document.getElementById('webcam-view');

  uploadBtn.addEventListener('click', () => {
    if (currentMode === 'upload') return;
    currentMode = 'upload';
    uploadBtn.classList.add('btn-primary'); uploadBtn.classList.remove('btn-secondary');
    webcamBtn.classList.add('btn-secondary'); webcamBtn.classList.remove('btn-primary');
    uploadView.classList.remove('hidden');
    webcamView.classList.add('hidden');
    if (webcamCtrl && webcamCtrl.isActive) webcamCtrl.stop();
    resetPredictionState();
  });

  webcamBtn.addEventListener('click', async () => {
    if (currentMode === 'webcam') return;
    currentMode = 'webcam';
    webcamBtn.classList.add('btn-primary'); webcamBtn.classList.remove('btn-secondary');
    uploadBtn.classList.add('btn-secondary'); uploadBtn.classList.remove('btn-primary');
    webcamView.classList.remove('hidden');
    uploadView.classList.add('hidden');
    resetPredictionState();
    await startWebcam();
  });
}

function initUpload() {
  const dropzone = document.getElementById('upload-dropzone');
  const fileInput = document.getElementById('file-input');
  const browseBtn = document.getElementById('browse-btn');

  browseBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    fileInput.click();
  });
  dropzone.addEventListener('click', (e) => {
    if (e.target === browseBtn || browseBtn.contains(e.target)) return;
    fileInput.click();
  });

  fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) handleFileSelected(file);
  });

  ['dragenter', 'dragover'].forEach(evt => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault(); e.stopPropagation();
      dropzone.classList.add('dropzone-active');
    });
  });
  ['dragleave', 'drop'].forEach(evt => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault(); e.stopPropagation();
      dropzone.classList.remove('dropzone-active');
    });
  });
  dropzone.addEventListener('drop', (e) => {
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) handleFileSelected(file);
  });
}

function handleFileSelected(file) {
  if (!file.type.startsWith('image/')) {
    showError('Please select an image file');
    return;
  }
  if (file.size > 25 * 1024 * 1024) {
    showError('File too large (max 25 MB)');
    return;
  }

  const reader = new FileReader();
  reader.onload = (e) => {
    lastCapturedDataUrl = e.target.result;
    lastCapturedBlob = file;
    showPreview(e.target.result, file.name, file.size);
    document.getElementById('predict-btn').disabled = false;
  };
  reader.readAsDataURL(file);
}

function showPreview(dataUrl, filename, sizeBytes) {
  const previewArea = document.getElementById('preview-area');
  const previewImg = document.getElementById('preview-img');
  const previewMeta = document.getElementById('preview-meta');

  previewImg.src = dataUrl;
  previewMeta.textContent = filename ? `${filename} · ${(sizeBytes / 1024).toFixed(1)} KB` : 'Webcam capture';
  previewArea.classList.remove('hidden');
  document.getElementById('upload-dropzone').classList.add('hidden');
}

function initWebcam() {
  const videoEl = document.getElementById('webcam-video');
  const canvasEl = document.getElementById('webcam-canvas');
  webcamCtrl = new WebcamController(videoEl, canvasEl);

  document.getElementById('capture-btn').addEventListener('click', () => {
    try {
      const dataUrl = webcamCtrl.capture();
      lastCapturedDataUrl = dataUrl;
      lastCapturedBlob = dataUrlToBlob(dataUrl);
      showPreview(dataUrl, null, lastCapturedBlob.size);
      document.getElementById('webcam-live').classList.add('hidden');
      document.getElementById('predict-btn').disabled = false;
    } catch (err) {
      showError(err.message);
    }
  });

  document.getElementById('retake-btn').addEventListener('click', async () => {
    resetPredictionState();
    document.getElementById('webcam-live').classList.remove('hidden');
    if (!webcamCtrl.isActive) await startWebcam();
  });
}

async function startWebcam() {
  const errorBox = document.getElementById('webcam-error');
  errorBox.classList.add('hidden');
  if (!webcamCtrl.isSupported()) {
    errorBox.textContent = 'Your browser does not support webcam access. Please use Chrome, Edge, or Firefox.';
    errorBox.classList.remove('hidden');
    return;
  }
  try {
    await webcamCtrl.start();
  } catch (err) {
    errorBox.textContent = err.message;
    errorBox.classList.remove('hidden');
  }
}

function initPredictButton() {
  document.getElementById('predict-btn').addEventListener('click', async () => {
    if (!lastCapturedBlob) {
      showError('No image selected');
      return;
    }
    await runPrediction();
  });

  document.getElementById('reset-btn').addEventListener('click', () => {
    resetPredictionState(true);
  });
}

function resetPredictionState(resetInputToo = false) {
  document.getElementById('results-section').classList.add('hidden');
  document.getElementById('error-banner').classList.add('hidden');

  if (resetInputToo) {
    lastCapturedBlob = null;
    lastCapturedDataUrl = null;
    document.getElementById('predict-btn').disabled = true;
    document.getElementById('preview-area').classList.add('hidden');
    document.getElementById('upload-dropzone').classList.remove('hidden');
    document.getElementById('file-input').value = '';
    if (currentMode === 'webcam') {
      document.getElementById('webcam-live').classList.remove('hidden');
    }
  }
}

async function runPrediction() {
  const predictBtn = document.getElementById('predict-btn');
  const loadingBox = document.getElementById('loading-state');
  const resultsSection = document.getElementById('results-section');

  predictBtn.disabled = true;
  predictBtn.querySelector('.btn-label').textContent = 'Analyzing...';
  predictBtn.querySelector('.btn-spinner').classList.remove('hidden');
  loadingBox.classList.remove('hidden');
  resultsSection.classList.add('hidden');
  document.getElementById('error-banner').classList.add('hidden');

  try {
    const response = await predictImage(lastCapturedBlob);
    renderResults(response);
  } catch (err) {
    showError(`Prediction failed: ${err.message}`);
  } finally {
    predictBtn.disabled = false;
    predictBtn.querySelector('.btn-label').textContent = 'Run Prediction';
    predictBtn.querySelector('.btn-spinner').classList.add('hidden');
    loadingBox.classList.add('hidden');
  }
}

function renderResults(response) {
  const { results, total_request_ms, preprocess_ms, device } = response;
  document.getElementById('total-time').textContent = `${total_request_ms.toFixed(0)} ms`;
  document.getElementById('preprocess-time').textContent = `${preprocess_ms.toFixed(0)} ms`;
  document.getElementById('inference-device').textContent = device.toUpperCase();

  const grid = document.getElementById('models-grid');
  grid.innerHTML = '';

  const modelOrder = ['EffNet+SwinT', 'DBA-ViNet', 'MobileNetV4'];
  modelOrder.forEach((modelName, idx) => {
    const r = results[modelName];
    if (!r) return;
    const card = renderModelCard(modelName, r, idx);
    grid.appendChild(card);
  });

  const consensusBox = renderConsensus(results);
  document.getElementById('consensus-box').innerHTML = consensusBox;

  document.getElementById('results-section').classList.remove('hidden');
  document.getElementById('results-section').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function renderModelCard(modelName, result, idx) {
  const colors = MODEL_COLORS[modelName];
  const desc = MODEL_DESCRIPTIONS[modelName];
  const meta = result.model_meta;
  const primary = result.primary;
  const secondary = result.secondary;

  const card = document.createElement('div');
  card.className = 'model-card glass-card p-5 fade-in';
  card.style.setProperty('--accent', colors.primary);
  card.style.animationDelay = `${idx * 0.1}s`;

  const primaryHtml = renderPrediction(primary, true, colors);
  const secondaryHtml = secondary ? renderPrediction(secondary, false, colors) : '';

  const gradcamHtml = primary.is_low_confidence ? '' : `
    <div class="mt-4">
      <div class="flex items-center gap-2 mb-2">
        <div class="text-xs font-mono uppercase tracking-wider text-gray-500">Grad-CAM Heatmap</div>
        <div class="tooltip-trigger" data-tip="Red areas show what the model focused on most. Cool colors mean low attention.">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="w-3.5 h-3.5 text-gray-500 cursor-help"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/></svg>
        </div>
      </div>
      <img src="${result.gradcam_image}" alt="Grad-CAM" class="rounded-lg w-full border border-gray-700" />
    </div>
  `;

  card.innerHTML = `
    <div class="flex items-start justify-between mb-4">
      <div>
        <div class="text-xs font-mono uppercase tracking-wider mb-1" style="color:${colors.primary}">
          MODEL ${String.fromCharCode(65 + idx)}
        </div>
        <h3 class="text-xl font-bold">${modelName}</h3>
        <p class="text-xs text-gray-500 mt-1">${desc}</p>
      </div>
      <div class="text-right">
        <div class="text-xs text-gray-500 mb-1">Inference</div>
        <div class="font-mono text-sm font-bold" style="color:${colors.primary}">${result.inference_ms.toFixed(0)} ms</div>
      </div>
    </div>

    ${primaryHtml}
    ${secondaryHtml}
    ${gradcamHtml}

    <div class="mt-4 pt-3 border-t border-gray-800 flex items-center justify-between text-xs text-gray-500">
      <div>Test acc: <span class="font-mono text-gray-300">${(meta.test_acc * 100).toFixed(1)}%</span></div>
      <div>Focus lift: <span class="font-mono text-gray-300">${meta.focus_lift.toFixed(2)}×</span></div>
    </div>
  `;
  return card;
}

function renderPrediction(pred, isPrimary, colors) {
  const isLow = pred.is_low_confidence;
  const conf = pred.confidence_pct;
  const label = pred.label;

  if (isLow) {
    return `
      <div class="${isPrimary ? 'prediction-primary' : 'prediction-secondary'} mb-3" style="border-color:#3a4655">
        <div class="flex items-center justify-between mb-2">
          <div class="text-xs font-mono uppercase tracking-wider text-gray-500">${isPrimary ? 'Top prediction' : 'Second prediction'}</div>
          <div class="font-mono text-xs text-gray-500">${conf.toFixed(1)}%</div>
        </div>
        <div class="flex items-center gap-2">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="w-4 h-4 text-gray-500 flex-shrink-0"><circle cx="12" cy="12" r="10"/><line x1="4.93" y1="4.93" x2="19.07" y2="19.07"/></svg>
          <div class="text-base font-bold text-gray-400 italic">Object Not Detected</div>
        </div>
        <div class="text-xs text-gray-600 mt-1 leading-relaxed">
          Confidence below 35% threshold — model is uncertain.
        </div>
      </div>
    `;
  }

  const barWidth = Math.max(2, conf);
  return `
    <div class="${isPrimary ? 'prediction-primary' : 'prediction-secondary'} mb-3" style="border-color:${colors.primary}; background: ${isPrimary ? colors.dim : 'transparent'};">
      <div class="flex items-center justify-between mb-2">
        <div class="text-xs font-mono uppercase tracking-wider" style="color:${colors.primary}">
          ${isPrimary ? 'Top Prediction' : 'Second Prediction'}
        </div>
        <div class="font-mono text-xs font-bold" style="color:${colors.primary}">
          ${conf.toFixed(1)}%
        </div>
      </div>
      <div class="${isPrimary ? 'text-xl' : 'text-base'} font-bold mb-2 leading-tight">${label}</div>
      <div class="confidence-bar-bg">
        <div class="confidence-bar-fill" style="width:${barWidth}%; background:${colors.primary};"></div>
      </div>
    </div>
  `;
}

function renderConsensus(results) {
  const labels = Object.values(results).map(r => r.primary.is_low_confidence ? 'NONE' : r.primary.raw_label);
  const allDetected = labels.every(l => l !== 'NONE');
  const uniqueLabels = [...new Set(labels)];

  if (!allDetected) {
    const detectedCount = labels.filter(l => l !== 'NONE').length;
    return `
      <div class="flex items-start gap-3">
        <div class="w-10 h-10 rounded-full bg-yellow-500/20 border border-yellow-500/40 flex items-center justify-center flex-shrink-0">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="w-5 h-5 text-yellow-400"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
        </div>
        <div>
          <div class="font-bold text-yellow-400">Partial Detection</div>
          <div class="text-sm text-gray-400 mt-1">${detectedCount} of 3 models confidently identified the fruit. Try a clearer image with better lighting.</div>
        </div>
      </div>
    `;
  }

  if (uniqueLabels.length === 1) {
    const fruitName = uniqueLabels[0].replace('_', ' — ');
    const trustLabel = MODEL_DESCRIPTIONS['MobileNetV4'].includes('Best overall') ? 'Highest-trust model (MobileNetV4) confirms this' : '';
    return `
      <div class="flex items-start gap-3">
        <div class="w-10 h-10 rounded-full bg-green-500/20 border border-green-500/40 flex items-center justify-center flex-shrink-0">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="w-5 h-5 text-green-400"><polyline points="20 6 9 17 4 12"/></svg>
        </div>
        <div>
          <div class="font-bold text-green-400">All 3 Models Agree</div>
          <div class="text-sm text-gray-300 mt-1">Verdict: <span class="font-bold">${fruitName}</span></div>
          <div class="text-xs text-gray-500 mt-1">${trustLabel}</div>
        </div>
      </div>
    `;
  }

  return `
    <div class="flex items-start gap-3">
      <div class="w-10 h-10 rounded-full bg-orange-500/20 border border-orange-500/40 flex items-center justify-center flex-shrink-0">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="w-5 h-5 text-orange-400"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
      </div>
      <div class="flex-grow">
        <div class="font-bold text-orange-400">Models Disagree</div>
        <div class="text-sm text-gray-400 mt-1">Predictions vary across models — this often happens at the Fresh ↔ Formalin boundary, where chemical adulteration is hardest to detect.</div>
        <div class="text-xs text-gray-500 mt-2">Trust hierarchy: <span class="font-mono text-green-400">MobileNetV4 (4.62× focus)</span> > <span class="font-mono text-orange-400">DBA-ViNet (2.82×)</span> > <span class="font-mono text-blue-400">EffNet+SwinT (1.46×)</span></div>
      </div>
    </div>
  `;
}

function showError(msg) {
  const banner = document.getElementById('error-banner');
  banner.querySelector('.error-text').textContent = msg;
  banner.classList.remove('hidden');
  banner.scrollIntoView({ behavior: 'smooth', block: 'center' });
}