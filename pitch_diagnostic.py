"""
Pitch Detection Diagnostic Tool

Use this to see what frequencies are being detected when you pluck each string.
Helps debug octave errors and harmonic confusion.
"""
import argparse
import numpy as np
import sounddevice as sd
import librosa
from scipy import signal

SR = 22050
CHUNK = 4096

# Reference frequencies for standard tuning
STANDARD_TUNING = {
    "E2 (6th string)": 82.41,
    "A2 (5th string)": 110.00,
    "D3 (4th string)": 146.83,
    "G3 (3rd string)": 196.00,
    "B3 (2nd string)": 246.94,
    "E4 (1st string)": 329.63,
}

def detect_pitch_detailed(y: np.ndarray, sr: int):
    """Detect pitch with detailed diagnostics"""
    # Remove DC offset
    y = y - np.mean(y)
    
    # High-pass filter
    b, a = signal.butter(2, 60 / (sr / 2), btype='high')
    y_filtered = signal.filtfilt(b, a, y)
    
    # YIN detection
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y_filtered,
        fmin=librosa.note_to_hz('E2') - 10,
        fmax=librosa.note_to_hz('E4') + 50,
        sr=sr,
        frame_length=4096,
        hop_length=512,
        win_length=4096,
    )
    
    valid_f0 = f0[~np.isnan(f0)]
    valid_probs = voiced_probs[~np.isnan(f0)]
    
    if len(valid_f0) == 0:
        return None, None, None, None
    
    freq_raw = float(np.median(valid_f0))
    conf = float(np.mean(valid_probs))
    
    # Octave correction
    freq_corrected = freq_raw
    octave_corrected = False
    if freq_raw > 200:
        test_freq = freq_raw / 2.0
        if 70 <= test_freq <= 250:
            freq_corrected = test_freq
            octave_corrected = True
    
    return freq_raw, freq_corrected, conf, octave_corrected


def find_closest_note(freq):
    """Find which standard tuning note this is closest to"""
    if freq is None or freq < 50:
        return None, 999
    
    closest = None
    min_error = float('inf')
    
    for note, target_freq in STANDARD_TUNING.items():
        error = abs(freq - target_freq)
        if error < min_error:
            min_error = error
            closest = note
    
    return closest, min_error


def main():
    parser = argparse.ArgumentParser(description="Diagnose guitar pitch detection")
    parser.add_argument("--device", type=int, default=None)
    args = parser.parse_args()
    
    print("=" * 70)
    print("PITCH DETECTION DIAGNOSTIC")
    print("=" * 70)
    print("\nThis tool shows what frequencies are detected when you pluck.")
    print("Use it to debug octave errors and tuning issues.\n")
    
    print("Standard tuning reference:")
    for note, freq in STANDARD_TUNING.items():
        print(f"  {note:20} = {freq:>6.2f} Hz")
    print()
    
    print("Instructions:")
    print("1. Pluck a string and hold for 2 seconds")
    print("2. Watch the detected frequency")
    print("3. Compare to expected frequency above")
    print("4. Press Ctrl+C when done\n")
    print("-" * 70)
    
    try:
        with sd.InputStream(
            samplerate=SR,
            blocksize=CHUNK,
            channels=1,
            dtype="float32",
            device=args.device,
        ) as stream:
            while True:
                input("\nPress ENTER, then pluck a string...")
                
                # Record 2 seconds
                frames = []
                for _ in range(int(2.0 * SR / CHUNK)):
                    data, _ = stream.read(CHUNK)
                    frames.append(data[:, 0])
                
                y = np.concatenate(frames)
                
                freq_raw, freq_corrected, conf, was_corrected = detect_pitch_detailed(y, SR)
                
                if freq_raw is None:
                    print("✗ No pitch detected!")
                    continue
                
                print(f"\n📊 Detection Results:")
                print(f"  Raw frequency:        {freq_raw:>7.1f} Hz")
                
                if was_corrected:
                    print(f"  Octave corrected to:  {freq_corrected:>7.1f} Hz  ← USED (was 1 octave up)")
                else:
                    print(f"  Corrected frequency:  {freq_corrected:>7.1f} Hz  (no correction needed)")
                
                print(f"  Confidence:           {conf:>7.2f}")
                
                closest, error = find_closest_note(freq_corrected)
                if closest:
                    tuning_status = "✓ In tune" if error < 5 else "⚠ Out of tune" if error < 15 else "✗ Very out of tune"
                    print(f"\n  Closest match:        {closest}")
                    print(f"  Error:                {error:>7.1f} Hz  {tuning_status}")
                
                # Show what note this actually is
                note_name = librosa.hz_to_note(freq_corrected)
                print(f"  Note name:            {note_name}")
                
                print("-" * 70)
    
    except KeyboardInterrupt:
        print("\n\nDiagnostic complete!")
        print("\n💡 Tips:")
        print("  - If 'Octave corrected' shows often, harmonics are too loud")
        print("  - If error > 15 Hz, your guitar needs tuning")
        print("  - If confidence < 0.5, pluck louder or move mic closer")


if __name__ == "__main__":
    main()