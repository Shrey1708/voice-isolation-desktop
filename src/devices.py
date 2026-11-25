"""
Audio device enumeration & virtual cable detection.

Lists all PortAudio input/output devices and identifies candidates
for the physical microphone and virtual audio cable.

Run directly:  python -m src.devices
"""

import sounddevice as sd

# Known virtual cable device name patterns (case-insensitive substrings)
VIRTUAL_CABLE_PATTERNS = [
    "cable input",    # VB-Cable playback endpoint (we write TO this)
    "cable output",   # VB-Cable capture endpoint (apps read FROM this)
    "blackhole",      # macOS BlackHole
    "virtual",        # Generic virtual device keyword
]


def get_all_devices():
    """Return the full list of PortAudio devices as dicts."""
    return sd.query_devices()


def classify_device(device):
    """Check if a device name matches a known virtual cable pattern."""
    name_lower = device["name"].lower()
    for pattern in VIRTUAL_CABLE_PATTERNS:
        if pattern in name_lower:
            return pattern
    return None


def find_devices():
    """
    Scan all audio devices and return categorized results.

    Returns a dict with:
        - inputs:  list of input devices (physical mics + virtual captures)
        - outputs: list of output devices (speakers + virtual playback)
        - virtual_inputs:  virtual cable capture endpoints (apps use as mic)
        - virtual_outputs: virtual cable playback endpoints (we write to)
        - physical_inputs: non-virtual input devices (physical mics)
    """
    all_devices = get_all_devices()

    inputs = []
    outputs = []
    virtual_inputs = []
    virtual_outputs = []
    physical_inputs = []

    for idx, dev in enumerate(all_devices):
        info = {
            "index": idx,
            "name": dev["name"],
            "host_api": sd.query_hostapis(dev["hostapi"])["name"],
            "max_input_channels": dev["max_input_channels"],
            "max_output_channels": dev["max_output_channels"],
            "default_samplerate": dev["default_samplerate"],
        }

        cable_match = classify_device(dev)
        is_virtual = cable_match is not None

        # Categorize inputs (devices that can capture audio)
        if dev["max_input_channels"] > 0:
            inputs.append(info)
            if is_virtual:
                virtual_inputs.append(info)
            else:
                physical_inputs.append(info)

        # Categorize outputs (devices that can play audio)
        if dev["max_output_channels"] > 0:
            outputs.append(info)
            if is_virtual:
                virtual_outputs.append(info)

    return {
        "inputs": inputs,
        "outputs": outputs,
        "virtual_inputs": virtual_inputs,
        "virtual_outputs": virtual_outputs,
        "physical_inputs": physical_inputs,
    }


def _pick_wasapi_48k(devices: list[dict], preferred_name: str | None = None) -> dict | None:
    """Prefer a WASAPI 48 kHz device, optionally matching preferred_name."""
    if preferred_name:
        target = preferred_name.lower()
        for dev in devices:
            if target in dev["name"].lower() and dev["host_api"] == "Windows WASAPI":
                return dev
        for dev in devices:
            if target in dev["name"].lower():
                return dev

    for dev in devices:
        if dev["host_api"] == "Windows WASAPI" and dev["default_samplerate"] == 48000:
            return dev

    return devices[0] if devices else None


def select_physical_mic(preferred_name: str | None = None) -> dict | None:
    """Pick the best physical microphone (preferring WASAPI at 48 kHz)."""
    result = find_devices()
    mics = [
        d for d in result["physical_inputs"]
        if "mapper" not in d["name"].lower() and "primary" not in d["name"].lower()
    ] or result["physical_inputs"]
    return _pick_wasapi_48k(mics, preferred_name)


def select_virtual_cable(preferred_name: str | None = None) -> dict | None:
    """Pick the virtual cable playback endpoint (e.g., CABLE Input at 48 kHz)."""
    result = find_devices()
    cables = [
        d for d in result["virtual_outputs"]
        if "cable input" in d["name"].lower() and "16ch" not in d["name"].lower()
    ] or result["virtual_outputs"]
    return _pick_wasapi_48k(cables, preferred_name)


def print_device_table(devices, label):
    """Pretty-print a list of devices."""
    print(f"\n{'=' * 70}")
    print(f"  {label} ({len(devices)} found)")
    print(f"{'=' * 70}")

    if not devices:
        print("  (none)")
        return

    for dev in devices:
        print(f"  [{dev['index']:>2}] {dev['name']}")
        print(f"       Host API: {dev['host_api']}  |  "
              f"In: {dev['max_input_channels']}ch  |  "
              f"Out: {dev['max_output_channels']}ch  |  "
              f"Rate: {int(dev['default_samplerate'])} Hz")


def main():
    """List all devices and highlight virtual cable + physical mic candidates."""
    print("Scanning PortAudio devices...\n")

    result = find_devices()

    print_device_table(result["physical_inputs"], "Physical Microphones")
    print_device_table(result["virtual_inputs"], "Virtual Cable Inputs (apps select as mic)")
    print_device_table(result["virtual_outputs"], "Virtual Cable Outputs (engine writes to)")
    print_device_table(result["outputs"], "All Output Devices")

    # Summary
    print(f"\n{'=' * 70}")
    print("  SUMMARY")
    print(f"{'=' * 70}")

    if result["physical_inputs"]:
        mic = result["physical_inputs"][0]
        print(f"  ✓ Physical mic candidate:  [{mic['index']}] {mic['name']}")
    else:
        print("  ✗ No physical microphone found!")

    if result["virtual_outputs"]:
        vout = result["virtual_outputs"][0]
        print(f"  ✓ Virtual cable output:    [{vout['index']}] {vout['name']}")
    else:
        print("  ✗ No virtual cable output found — install VB-Cable!")

    if result["virtual_inputs"]:
        vin = result["virtual_inputs"][0]
        print(f"  ✓ Virtual cable input:     [{vin['index']}] {vin['name']}")
    else:
        print("  ✗ No virtual cable input found — install VB-Cable!")


if __name__ == "__main__":
    main()
