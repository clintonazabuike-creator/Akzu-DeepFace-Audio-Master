# Akzu-DeepFace-Audio-Master

A high-performance, real-time AI video and audio deepfake suite optimized for **NVIDIA RTX hardware** (CUDA execution providers). Designed for real-time upper-body face swapping, CIELAB skin-tone harmonization, and sub-50ms RVC voice conversion routed through OBS Virtual Camera.

---

## 🏛️ System Architecture & Stack

- **Computer Vision:** Deep-Live-Cam / LivePortrait core modules running on ONNX Runtime GPU (CUDA).
- **Color Harmonization:** Custom `color_matcher.py` executing CIELAB color space mean-std normalization to eliminate face/neck boundary artifacts.
- **Audio Synthesis:** Real-time Retrieval-based Voice Conversion (`run_rvc_live.py`) with chunked audio stream buffers (`<50ms latency`).
- **Virtual Production:** OBS Studio integration via Virtual Camera and VB-Audio Virtual Cable for zero-lag communication app injection (Zoom, WhatsApp, Telegram).

---

## 📁 Repository Directory Structure

```text
deepfake-voice-video-suite/
├── README.md                 # Project documentation & execution guide
├── requirements.txt          # Pinned dependencies (strict numpy<2.0.0)
├── setup.bat                 # Automated Windows VENV & CUDA installer
├── run_suite.bat             # Master orchestration script
├── video_pipeline/
│   ├── run_video.py          # CUDA video inference script
│   ├── color_matcher.py      # CIELAB skin-tone matcher
│   └── models/               # Ignored weights directory (.onnx / source images)
├── voice_pipeline/
│   ├── run_rvc_live.py       # Real-time RVC audio streaming script
│   └── trained_voices/       # Ignored model weights directory (.pth / .index)
└── configs/
    └── obs_settings.json     # OBS profile & virtual routing parameters
# Akzu-DeepFace-Audio-Master
