const API_BASE = window.location.origin;

async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/api/health`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch (err) {
    return { status: 'error', error: err.message };
  }
}

async function getModelsInfo() {
  const res = await fetch(`${API_BASE}/api/models`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return await res.json();
}

async function predictImage(imageBlob, filename = 'capture.png') {
  const formData = new FormData();
  formData.append('file', imageBlob, filename);

  const res = await fetch(`${API_BASE}/api/predict`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    let errorMsg = `Server returned ${res.status}`;
    try {
      const err = await res.json();
      if (err.detail) errorMsg = err.detail;
    } catch {}
    throw new Error(errorMsg);
  }

  return await res.json();
}

function dataUrlToBlob(dataUrl) {
  const [header, data] = dataUrl.split(',');
  const mime = header.match(/:(.*?);/)[1];
  const binary = atob(data);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  return new Blob([bytes], { type: mime });
}