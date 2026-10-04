/**
 * Push-to-Talk and Audio Playback Subsystem
 */
class VoiceController {
  constructor() {
    this.mediaRecorder = null;
    this.audioChunks = [];
    this.isRecording = false;
    this.currentAudioPlayer = null;
  }

  async startRecording() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error(
        "Microphone access unavailable. Ensure you are on HTTPS or localhost (W3C Secure Context)."
      );
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: 16000,
          echoCancellation: true,
          noiseSuppression: true,
        },
      });

      this.audioChunks = [];
      const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus"
        : "audio/mp4";

      this.mediaRecorder = new MediaRecorder(stream, { mimeType });
      this.mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          this.audioChunks.push(event.data);
        }
      };

      this.mediaRecorder.start();
      this.isRecording = true;
    } catch (err) {
      if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
        throw new Error("Microphone permission denied by browser.");
      }
      throw err;
    }
  }

  stopRecording() {
    return new Promise((resolve, reject) => {
      if (!this.mediaRecorder || this.mediaRecorder.state === "inactive") {
        resolve(null);
        return;
      }

      this.mediaRecorder.onstop = () => {
        this.isRecording = false;
        // Stop audio tracks
        this.mediaRecorder.stream.getTracks().forEach((t) => t.stop());

        const blob = new Blob(this.audioChunks, { type: this.mediaRecorder.mimeType });
        this.audioChunks = [];
        resolve(blob);
      };

      this.mediaRecorder.onerror = (e) => reject(e);
      this.mediaRecorder.stop();
    });
  }

  async playAudioBlob(blob) {
    if (this.currentAudioPlayer) {
      this.currentAudioPlayer.pause();
    }
    const url = URL.createObjectURL(blob);
    const audio = new Audio(url);
    this.currentAudioPlayer = audio;
    audio.onended = () => URL.revokeObjectURL(url);
    await audio.play();
  }
}

window.Voice = new VoiceController();
