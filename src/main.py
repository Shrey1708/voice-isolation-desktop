"""
Main entry point for the real-time voice isolation engine.

Routes physical mic audio -> 48k->16k resampler -> VoxLock TSE -> 16k->48k
resampler -> VB-Cable output for Google Meet, Discord, Zoom, etc.

Usage:
    python -m src.main                # live TSE isolation (default)
    python -m src.main --passthrough  # baseline passthrough without ML
    python -m src.main --list         # list audio devices and exit

Controls (while running):
    b + Enter  - toggle Isolation vs. Bypass
    r + Enter  - reset stream state
    g + Enter  - toggle optional noise gate (TSE mode)
    q + Enter  - quit
"""

import argparse
import json
import sys
import threading
from pathlib import Path

from .audio_io import AudioEngine
from .devices import find_devices, print_device_table, select_physical_mic, select_virtual_cable
from .processor_base import PassthroughProcessor
from .tse_voxlock import VoxLockProcessor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.json"


def load_config() -> dict:
    """Load saved config or return defaults."""
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def select_devices(config: dict | None = None) -> tuple[int, int]:
    """Auto-detect the physical mic and virtual cable output."""
    cfg = config or {}
    mic = select_physical_mic(cfg.get("input_device"))
    if mic is None:
        print("ERROR: No physical microphone found!")
        sys.exit(1)
    print(f"  Input (physical mic):   [{mic['index']}] {mic['name']} ({mic['host_api']})")

    cable = select_virtual_cable(cfg.get("output_device"))
    if cable is None:
        print("ERROR: No virtual cable output found! Install VB-Cable.")
        sys.exit(1)
    print(f"  Output (virtual cable): [{cable['index']}] {cable['name']} ({cable['host_api']})")

    return mic["index"], cable["index"]


def keyboard_listener(engine: AudioEngine, stop_event: threading.Event) -> None:
    """Listen for live keyboard commands on stdin."""
    has_gate = isinstance(engine.processor, VoxLockProcessor)
    gate_hint = "  |  [g] toggle noise gate" if has_gate else ""
    print(f"\nControls:  [b] toggle bypass  |  [r] reset state{gate_hint}  |  [q] quit\n")

    while not stop_event.is_set():
        try:
            key = input().strip().lower()
        except EOFError:
            break

        if key == "b":
            engine.bypass = not engine.bypass
            state = "BYPASS" if engine.bypass else "ISOLATING"
            print(f"  -> Mode: {state}")
        elif key == "r":
            engine.reset()
            print("  -> Stream state reset")
        elif key == "g" and has_gate:
            engine.processor.noise_gate = not engine.processor.noise_gate
            state = "ON" if engine.processor.noise_gate else "OFF"
            print(f"  -> Noise gate: {state}")
        elif key == "q":
            stop_event.set()


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-Time Voice Isolation Engine")
    parser.add_argument("--list", action="store_true", help="List audio devices and exit")
    parser.add_argument("--passthrough", action="store_true", help="Run without TSE model")
    parser.add_argument("--noise-gate", action="store_true", help="Enable optional post-TSE noise gate")
    args = parser.parse_args()

    if args.list:
        result = find_devices()
        print_device_table(result["physical_inputs"], "Physical Microphones")
        print_device_table(result["virtual_outputs"], "Virtual Cable Outputs")
        print_device_table(result["virtual_inputs"], "Virtual Cable Inputs")
        return

    config = load_config()
    chunk_ms = config.get("chunk_ms", 40)
    sample_rate = config.get("sample_rate", 48000)

    print("=== Real-Time Voice Isolation Engine ===\n")
    print("Selecting audio devices...")
    input_idx, output_idx = select_devices(config)

    if args.passthrough:
        processor = PassthroughProcessor()
        mode_label = "Passthrough (no AI)"
    else:
        vp_path = PROJECT_ROOT / config.get("voiceprint_path", "data/voiceprint.wav")
        if not vp_path.exists():
            print(f"\nERROR: Voiceprint not found at {vp_path}")
            print("Run `python -m src.enrollment` first, or use `--passthrough`.")
            sys.exit(1)

        threads = config.get("onnx_threads", 2)
        gate_enabled = args.noise_gate or config.get("noise_gate", False)
        gate_db = config.get("gate_threshold_db", -45.0)
        print(f"Loading VoxLock (voiceprint={vp_path.name}, threads={threads})...")
        processor = VoxLockProcessor(
            voiceprint_path=vp_path,
            threads=threads,
            noise_gate=gate_enabled,
            gate_threshold_db=gate_db,
        )
        mode_label = f"VoxLock TSE (noise_gate={'ON' if gate_enabled else 'OFF'})"

    engine = AudioEngine(
        input_device=input_idx,
        output_device=output_idx,
        processor=processor,
        chunk_ms=chunk_ms,
        sample_rate=sample_rate,
    )

    stop_event = threading.Event()

    try:
        engine.start()
        print(f"\n✓ Engine running [{mode_label}]. Select CABLE Output in Meet/Discord.")
        keyboard_listener(engine, stop_event)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        print("Stopping engine...")
        engine.stop()
        print("Done.")


if __name__ == "__main__":
    main()
