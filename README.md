# Real-Time Desktop Voice Isolation (Pure Python)

Real-time system-wide desktop voice isolation engine written in Python. Routes your microphone through a stateful `48 kHz <-> 16 kHz` streaming pipeline and [VoxLock](https://github.com/arbazsiddiqui/voxlock) Target Speaker Extraction (TSE) into a virtual microphone (**VB-Cable** / **BlackHole**) for **Discord**, **Google Meet**, and **Zoom**.

Unlike generic noise suppressors, TSE is conditioned on your enrolled voiceprint so it suppresses **both background noise and competing human voices**.

## Features

- **Target Speaker Extraction (`voxlock`, ONNX CPU):** Isolates only the enrolled speaker's voice (~24 ms algorithmic latency at 16 kHz).
- **Glitch-free stateful resampling (`soxr`):** Continuous `48 kHz -> 16 kHz -> 48 kHz` conversion without chunk-boundary clicks.
- **Decoupled real-time audio I/O (`sounddevice`):** Minimal audio callbacks backed by bounded queues and a dedicated worker thread.
- **Live controls & telemetry:** Toggle isolation/bypass (`b`), reset stream state (`r`), toggle optional post-TSE noise gate (`g`), and view rolling RTF / queue depth / XRUN stats every 5 s.

## Requirements

- Python 3.10+ (tested on Python 3.13 / Windows 11)
- Virtual audio driver: [VB-Cable](https://vb-audio.com/Cable/) (Windows) or BlackHole 2ch (macOS)

```bash
pip install -r requirements.txt
```

## Quick Start

1. **Verify audio devices & virtual cable:**
   ```bash
   python -m src.devices
   ```
2. **Record your clean 5–10 s voiceprint** (`data/voiceprint.wav`):
   ```bash
   python -m src.enrollment
   ```
3. **Verify offline isolation & CPU RTF:**
   ```bash
   python -m tests.test_offline_tse
   ```
4. **Run live voice isolation:**
   ```bash
   python -m src.main
   ```
   Select **CABLE Output** as your microphone in Discord / Google Meet (disable app-side noise suppression to avoid double-processing).
