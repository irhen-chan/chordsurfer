"""
Microphone diagnostic tool for ChordSurfer
Run this to verify your mic is working and at good levels
"""
import argparse
import time
import numpy as np
import sounddevice as sd

SR = 22050
CHUNK = 2048

def rms(x):
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))

def level_meter(level, width=40):
    """Visual level meter using blocks"""
    # Target range: 0.001 (silent) to 0.10 (loud strum)
    # Use log scale for better visualization
    if level < 0.0001:
        bars = 0
    else:
        # Map 0.001 to 0.10 onto 0 to width
        log_level = np.log10(level)
        # -3 (0.001) to -1 (0.10)
        normalized = (log_level + 3) / 2  # maps to 0-1
        bars = int(max(0, min(width, normalized * width)))
    
    meter = "█" * bars + "░" * (width - bars)
    return meter

def test_microphone(device=None, duration=10):
    """
    Real-time mic level display with pass/fail indicators
    """
    print("=" * 60)
    print("MICROPHONE DIAGNOSTIC TEST")
    print("=" * 60)
    
    if device is not None:
        dev_info = sd.query_devices(device)
        print(f"\nUsing device {device}: {dev_info['name']}")
    else:
        dev_info = sd.query_devices(kind='input')
        print(f"\nUsing default input: {dev_info['name']}")
    
    print(f"\nRecording for {duration} seconds...")
    print("Strum your guitar loudly several times!\n")
    print("TARGET LEVELS:")
    print("  Silent/Noise: 0.0001 - 0.001  (too quiet ❌)")
    print("  Good signal:  0.005 - 0.050   (perfect ✓)")
    print("  Loud strum:   0.050 - 0.200   (good ✓)")
    print("  Clipping:     > 0.500          (too loud ⚠️)\n")
    
    levels = []
    max_seen = 0.0
    
    def callback(indata, frames, time_info, status):
        nonlocal max_seen
        if status:
            print(status)
        
        mono = indata[:, 0]
        level = rms(mono)
        peak = float(np.max(np.abs(mono)))
        
        levels.append(level)
        max_seen = max(max_seen, level)
        
        # Visual meter
        meter = level_meter(level)
        
        # Status indicator
        if level < 0.001:
            status_icon = "❌"
            status_text = "TOO QUIET"
        elif level < 0.005:
            status_icon = "⚠️"
            status_text = "WEAK"
        elif level < 0.200:
            status_icon = "✓"
            status_text = "GOOD"
        else:
            status_icon = "⚠️"
            status_text = "TOO LOUD"
        
        print(f"{meter} {level:>7.4f} {status_icon} {status_text}  (peak: {peak:.4f})", end="\r")
    
    try:
        with sd.InputStream(
            samplerate=SR,
            blocksize=CHUNK,
            channels=1,
            dtype="float32",
            device=device,
            callback=callback,
        ):
            time.sleep(duration)
    
    except KeyboardInterrupt:
        print("\n\nTest interrupted.")
    
    print("\n")
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    if not levels:
        print("❌ No audio captured! Check microphone connection.")
        return False
    
    avg_level = np.mean(levels)
    max_level = max_seen
    p95 = np.percentile(levels, 95)
    
    print(f"\nAverage RMS:    {avg_level:.6f}")
    print(f"Max RMS:        {max_level:.6f}")
    print(f"95th percentile: {p95:.6f}")
    
    print("\nDIAGNOSIS:")
    
    # Check if basically silent
    if max_level < 0.002:
        print("❌ CRITICAL: Microphone is not picking up audio!")
        print("   Possible causes:")
        print("   - Wrong input device selected")
        print("   - Microphone is muted in Windows")
        print("   - Input volume is set to 0%")
        print("   - Microphone not plugged in properly")
        print("   - Hardware mute switch enabled")
        print("\n   ACTION: Check Windows Sound Settings (Input)")
        return False
    
    elif max_level < 0.005:
        print("⚠️  WARNING: Signal is very weak")
        print("   - Increase input volume in Windows (aim for 80-100%)")
        print("   - Move microphone closer to guitar (20-40cm)")
        print("   - If using audio interface, increase gain knob")
        print("\n   You might be able to calibrate, but it will be unreliable.")
        return False
    
    elif max_level < 0.020:
        print("⚠️  Marginal signal strength")
        print("   - Signal might work but could be unreliable")
        print("   - Try increasing input volume slightly")
        print("   - Consider moving mic closer")
        return True
    
    elif max_level < 0.200:
        print("✓ EXCELLENT: Signal strength is perfect!")
        print("   You're ready to run calibration.")
        return True
    
    else:
        print("⚠️  Signal is VERY loud (may be clipping)")
        print("   - Try reducing input volume in Windows")
        print("   - Move mic slightly further from guitar")
        print("   - But it should still work for calibration")
        return True


def main():
    parser = argparse.ArgumentParser(description="Test microphone input for ChordSurfer")
    parser.add_argument("--list-devices", action="store_true", help="List all audio devices")
    parser.add_argument("--device", type=int, default=None, help="Device index to test")
    parser.add_argument("--duration", type=int, default=10, help="Test duration in seconds")
    args = parser.parse_args()
    
    if args.list_devices:
        print("Available audio devices:")
        print(sd.query_devices())
        return
    
    success = test_microphone(device=args.device, duration=args.duration)
    
    if success:
        print("\n✓ Microphone test PASSED. Run calibration:")
        if args.device is not None:
            print(f"  python main.py --calibrate --device {args.device}")
        else:
            print("  python main.py --calibrate")
    else:
        print("\n❌ Microphone test FAILED. Fix the issues above, then try again.")


if __name__ == "__main__":
    main()