class WebcamController {
  constructor(videoElement, canvasElement) {
    this.video = videoElement;
    this.canvas = canvasElement;
    this.stream = null;
    this.isActive = false;
  }

  async start() {
    if (this.isActive) return;
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width:  { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: 'environment',
        },
        audio: false,
      });
      this.video.srcObject = this.stream;
      await this.video.play();
      this.isActive = true;
      return true;
    } catch (err) {
      console.error('Webcam error:', err);
      let userMsg = 'Could not access webcam.';
      if (err.name === 'NotAllowedError') {
        userMsg = 'Camera permission was denied. Please allow camera access in your browser settings.';
      } else if (err.name === 'NotFoundError') {
        userMsg = 'No camera detected on this device.';
      } else if (err.name === 'NotReadableError') {
        userMsg = 'Camera is in use by another application.';
      }
      throw new Error(userMsg);
    }
  }

  stop() {
    if (this.stream) {
      this.stream.getTracks().forEach(track => track.stop());
      this.stream = null;
    }
    this.video.srcObject = null;
    this.isActive = false;
  }

  capture() {
    if (!this.isActive || !this.video.videoWidth) {
      throw new Error('Webcam not ready yet');
    }
    const w = this.video.videoWidth;
    const h = this.video.videoHeight;
    this.canvas.width = w;
    this.canvas.height = h;
    const ctx = this.canvas.getContext('2d');
    ctx.drawImage(this.video, 0, 0, w, h);
    return this.canvas.toDataURL('image/png');
  }

  isSupported() {
    return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
  }
}