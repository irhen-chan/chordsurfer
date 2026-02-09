"""
ChordSurfer - Guitar to Keyboard Controller
============================================

A real-time audio-to-keyboard mapper that lets you play Subway Surfers
(or any game) using your guitar as a controller.

Detects individual guitar notes via FFT pitch detection and maps them to
arrow key presses, enabling guitar-based gameplay.

Author: Snehar
License: MIT
"""

import argparse
import json
import os
import queue
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import sounddevice as sd
from scipy import signal
from pynput.keyboard import Key, Controller


# ============================================================================
# Configuration
# ============================================================================

# Audio settings
SR = 22050              # Sample rate (Hz)
CHUNK = 4096            # Samples per chunk (~186ms at 22050 Hz)

# Onset detection thresholds
ONSET_RATIO = 1.8       # Multiplier above noise floor to trigger detection
ONSET_ABS = 0.004       # Absolute RMS minimum for onset detection

# Timing controls
COOLDOWN_SEC = 0.35         # Minimum time between any actions (seconds)
SAME_NOTE_COOLDOWN = 0.5    # Minimum time before same note can retrigger (seconds)

# Pitch detection settings
FREQ_TOLERANCE = 20     # Hz tolerance for note matching (±20 Hz)
CONFIDENCE_MIN = 0.3    # Minimum confidence threshold (0-1)
POST_ONSET_CHUNKS = 2   # Number of audio chunks to collect after onset

# File and system settings
TEMPLATES_FILE = "note_templates.json"  # Calibration data storage
MAX_QUEUE_SIZE = 50                     # Audio queue size limit


# ============================================================================
# Note Mapping Configuration
# ============================================================================

# Maps detected notes to keyboard keys
# Using higher-frequency notes (G3-E4) for better laptop mic detection
NOTE_TO_KEY = {
    "G3": Key.up,       # 3rd string open (~196 Hz) → Jump
    "B3": Key.down,     # 2nd string open (~247 Hz) → Roll/Duck
    "D4": Key.left,     # 2nd string 3rd fret (~294 Hz) → Move Left
    "E4": Key.right,    # 1st string open (~329 Hz) → Move Right
}

# Human-readable tab descriptions
NOTE_TABS = {
    "G3": "3rd string (G), open",
    "B3": "2nd string (B), open",
    "D4": "2nd string (B), 3rd fret",
    "E4": "1st string (high E), open",
}


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class Templates:
    """Stores calibrated frequency templates for each note.
    
    Attributes:
        frequencies: Dict mapping note names to their calibrated frequencies (Hz)
    """
    frequencies: Dict[str, float]


# ============================================================================
# Signal Processing Functions
# ============================================================================

def rms(x: np.ndarray) -> float:
    """Calculate Root Mean Square (RMS) amplitude of a signal.
    
    RMS provides a measure of signal power/loudness that's more perceptually
    accurate than simple peak amplitude.
    
    Args:
        x: Input signal array
        
    Returns:
        RMS amplitude as a float
    """
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))


def detect_pitch_fft(y: np.ndarray, sr: int) -> Tuple[float, float]:
    """Detect fundamental frequency using FFT-based peak detection.
    
    This approach is simpler and more robust than autocorrelation methods
    (like YIN) for noisy/weak signals from laptop microphones.
    
    Algorithm:
        1. Remove DC offset and apply Hanning window
        2. Compute FFT and find peak magnitude in guitar frequency range
        3. Calculate confidence based on peak-to-noise ratio
    
    Args:
        y: Audio signal samples
        sr: Sample rate (Hz)
        
    Returns:
        Tuple of (detected_frequency_hz, confidence_0_to_1)
        Returns (0.0, 0.0) if no clear pitch detected
    """
    # Remove DC offset (center signal around zero)
    y = y - np.mean(y)
    
    # Apply Hanning window to reduce spectral leakage
    window = np.hanning(len(y))
    y_windowed = y * window
    
    # Compute FFT (Fast Fourier Transform)
    fft = np.fft.rfft(y_windowed)
    freqs = np.fft.rfftfreq(len(y_windowed), 1/sr)
    magnitude = np.abs(fft)
    
    # Focus on guitar range (150-400 Hz for higher strings)
    # Below 150 Hz: noise, rumble, and low harmonics
    # Above 400 Hz: mostly harmonics, not fundamentals
    min_freq_idx = np.argmax(freqs >= 150)
    max_freq_idx = np.argmax(freqs >= 400)
    
    # Extract the search region
    search_mag = magnitude[min_freq_idx:max_freq_idx]
    search_freqs = freqs[min_freq_idx:max_freq_idx]
    
    if len(search_mag) == 0:
        return 0.0, 0.0
    
    # Find peak frequency (loudest frequency in range)
    peak_idx = np.argmax(search_mag)
    detected_freq = search_freqs[peak_idx]
    peak_magnitude = search_mag[peak_idx]
    
    # Calculate confidence based on peak-to-noise ratio
    # Higher peak relative to median = clearer signal = higher confidence
    noise_floor = np.median(search_mag)
    signal_to_noise = peak_magnitude / (noise_floor + 1e-9)
    confidence = min(1.0, signal_to_noise / 10.0)  # Scale to 0-1 range
    
    return float(detected_freq), float(confidence)


def match_note(freq: float, templates: Templates, 
               tolerance: float = FREQ_TOLERANCE) -> Tuple[Optional[str], float]:
    """Match detected frequency to closest calibrated note.
    
    Uses nearest-neighbor matching with a tolerance threshold to prevent
    false matches from noise or out-of-tune notes.
    
    Args:
        freq: Detected frequency in Hz
        templates: Calibrated note templates
        tolerance: Maximum Hz error to accept as a match
        
    Returns:
        Tuple of (matched_note_name, frequency_error_hz)
        Returns (None, error) if no match within tolerance
    """
    if freq < 100:  # Too low to be a valid guitar note in our range
        return None, 999.0
    
    # Find closest note by minimum absolute frequency difference
    best_note = None
    best_error = float('inf')
    
    for note, target_freq in templates.frequencies.items():
        error = abs(freq - target_freq)
        if error < best_error:
            best_error = error
            best_note = note
    
    # Only accept match if within tolerance
    if best_error <= tolerance:
        return best_note, best_error
    
    return None, best_error


# ============================================================================
# File I/O Functions
# ============================================================================

def save_templates(t: Templates, path: str = TEMPLATES_FILE) -> None:
    """Save calibrated note templates to JSON file.
    
    Args:
        t: Templates object to save
        path: Output file path (default: note_templates.json)
    """
    data = {k: v for k, v in t.frequencies.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_templates(path: str = TEMPLATES_FILE) -> Templates:
    """Load calibrated note templates from JSON file.
    
    Args:
        path: Input file path (default: note_templates.json)
        
    Returns:
        Templates object
        
    Raises:
        ValueError: If file format is invalid
        FileNotFoundError: If templates file doesn't exist
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if not isinstance(data, dict):
        raise ValueError(f"Invalid template file format")
    
    frequencies = {k: float(v) for k, v in data.items()}
    
    # Warn if any expected notes are missing
    missing = set(NOTE_TO_KEY.keys()) - set(frequencies.keys())
    if missing:
        print(f"[WARN] Templates missing notes: {missing}")
    
    return Templates(frequencies=frequencies)


# ============================================================================
# Audio Utility Functions
# ============================================================================

def list_devices() -> None:
    """Print available audio input devices."""
    print(sd.query_devices())


def flush_queue(q: queue.Queue) -> None:
    """Clear all pending items from audio queue.
    
    Used to prevent lag from stale audio data after note detection.
    
    Args:
        q: Queue to flush
    """
    try:
        while True:
            q.get_nowait()
    except queue.Empty:
        pass


def record_for_seconds(stream_q: queue.Queue, seconds: float) -> np.ndarray:
    """Record audio from queue for specified duration.
    
    Flushes queue first to ensure fresh recording.
    
    Args:
        stream_q: Audio input queue
        seconds: Duration to record
        
    Returns:
        Concatenated audio samples as numpy array
    """
    flush_queue(stream_q)  # Start with fresh data
    
    samples_needed = int(seconds * SR)
    buf = []
    got = 0
    
    while got < samples_needed:
        x = stream_q.get()
        buf.append(x)
        got += len(x)
    
    y = np.concatenate(buf)[:samples_needed]
    return y


# ============================================================================
# Calibration
# ============================================================================

def calibrate(templates_path: str, device: Optional[int], 
              notes: Tuple[str, ...]) -> None:
    """Interactive calibration to learn guitar's note frequencies.
    
    User plucks each note, system detects frequency, saves to template file.
    This accounts for guitar tuning variations and microphone characteristics.
    
    Args:
        templates_path: Output file path for calibration data
        device: Audio input device index (None = default)
        notes: Tuple of note names to calibrate
    """
    audio_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=MAX_QUEUE_SIZE)

    def callback(indata, frames, time_info, status):
        """Audio stream callback - runs in separate thread."""
        if status:
            print(status)
        try:
            audio_q.put_nowait(indata[:, 0].copy())  # Mono channel
        except queue.Full:
            pass  # Drop frames if queue is full

    # Print calibration instructions
    print("\n" + "=" * 70)
    print("CALIBRATION - FFT-Based Detection")
    print("=" * 70)
    print("\n🎸 USING HIGHER STRINGS (Easier to Detect!)")
    print("\nYou'll pluck 4 notes on the top 3 strings.")
    print("Higher frequencies work MUCH better with laptop mics!\n")
    print("Tips:")
    print("- Pluck FIRMLY (these need good volume)")
    print("- Let ring for 2 seconds")
    print("- Mic 20-30cm from guitar")
    print("- Windows input volume at 100%\n")

    with sd.InputStream(
        samplerate=SR,
        blocksize=CHUNK,
        channels=1,
        dtype="float32",
        device=device,
        callback=callback,
    ):
        # Measure background noise level
        print("Measuring background noise for 1.5s... stay quiet.")
        y_noise = record_for_seconds(audio_q, 1.5)
        noise_level = rms(y_noise)
        print(f"Noise baseline RMS ≈ {noise_level:.6f}\n")

        learned: Dict[str, float] = {}

        # Calibrate each note
        for note in notes:
            tab = NOTE_TABS[note]
            print("=" * 70)
            print(f"Note: {note}  →  {tab}")
            print("=" * 70)
            input("Press ENTER, then pluck FIRMLY and hold for ~2 seconds...")
            
            # Record note
            y = record_for_seconds(audio_q, 2.0)
            
            # Check if signal is strong enough
            max_rms = max([rms(y[i:i + CHUNK]) 
                          for i in range(0, len(y) - CHUNK, CHUNK)] or [0.0])
            
            if max_rms < 0.005:
                print(f"\n[ERROR] Signal too weak for {note}!")
                print(f"        Max RMS was only {max_rms:.6f} (need > 0.01)")
                print("        Pluck MUCH harder or increase input volume!")
                retry = input("Skip this note? (Y/n): ")
                if retry.lower() != 'n':
                    continue
            
            # Detect pitch
            freq, conf = detect_pitch_fft(y, SR)
            
            # If detection is weak, offer retry
            if freq < 100 or conf < 0.2:
                print(f"[WARN] Weak detection for {note}")
                print(f"       Detected: {freq:.1f} Hz (confidence: {conf:.2f})")
                print("       Try plucking harder!")
                input("Press ENTER to retry...")
                
                y2 = record_for_seconds(audio_q, 2.0)
                freq2, conf2 = detect_pitch_fft(y2, SR)
                
                # Use better attempt
                if conf2 > conf:
                    freq, conf = freq2, conf2
            
            if freq < 100:
                print(f"[ERROR] Failed to detect {note}. Skipping.")
                continue
            
            # Save calibrated frequency
            learned[note] = freq
            print(f"✓ Saved {note}: {freq:.1f} Hz (confidence: {conf:.2f})\n")

        # Check if calibration succeeded
        if not learned:
            print("[ERROR] No notes were learned.")
            print("\n💡 Your mic might not be picking up the guitar.")
            print("   Try: python mic_diagnostic.py")
            return

        # Save templates to file
        save_templates(Templates(frequencies=learned), templates_path)
        print("=" * 70)
        print(f"[DONE] Templates saved to: {templates_path}")
        print("=" * 70)
        print("\nYour calibrated notes:")
        for note, freq in learned.items():
            action = NOTE_TO_KEY[note]
            print(f"  {note}: {freq:.1f} Hz  →  {action}")
        print()


# ============================================================================
# Real-Time Detection Loop
# ============================================================================

def detect_loop(templates: Templates, device: Optional[int], 
                debug: bool = False) -> None:
    """Main detection loop - converts guitar notes to keyboard presses.
    
    Continuously monitors audio input, detects note onsets, performs pitch
    detection, matches to calibrated notes, and simulates keyboard presses.
    
    Features:
    - Onset detection to trigger only on fresh plucks
    - Same-note cooldown to prevent double-triggers from string decay
    - Adaptive noise floor tracking via EMA (Exponential Moving Average)
    - Queue flushing after detection to minimize latency
    
    Args:
        templates: Calibrated note templates
        device: Audio input device index (None = default)
        debug: If True, print detailed detection info
    """
    keyboard = Controller()
    audio_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=MAX_QUEUE_SIZE)

    def callback(indata, frames, time_info, status):
        """Audio stream callback - runs in separate thread."""
        if status:
            print(status)
        try:
            audio_q.put_nowait(indata[:, 0].copy())  # Mono channel
        except queue.Full:
            pass  # Drop frames if processing is too slow

    # Print gameplay instructions
    print("\n" + "=" * 70)
    print("PLAY MODE - FFT Detection")
    print("=" * 70)
    print("\n1) Open Subway Surfers in browser")
    print("2) Click inside game")
    print("3) Start plucking!\n")
    
    print("Note Mapping:")
    print("-" * 70)
    for note, key in NOTE_TO_KEY.items():
        tab = NOTE_TABS[note]
        freq = templates.frequencies.get(note, 0)
        print(f"  {note:>3} ({freq:>5.1f} Hz)  →  {str(key):12}  |  {tab}")
    print("-" * 70)
    print("\nPress Ctrl+C to stop.\n")

    def estimate_noise() -> float:
        """Estimate ambient noise floor over 1 second."""
        chunks = []
        start = time.monotonic()
        while time.monotonic() - start < 1.0:
            chunks.append(audio_q.get())
        vals = [rms(chunk) for chunk in chunks]
        return float(np.median(vals)) if vals else 0.001

    try:
        with sd.InputStream(
            samplerate=SR,
            blocksize=CHUNK,
            channels=1,
            dtype="float32",
            device=device,
            callback=callback,
        ):
            # Initial noise calibration
            print("Calibrating noise... stay quiet.")
            noise_floor = estimate_noise()
            if debug:
                print(f"[DEBUG] noise_floor={noise_floor:.6f}\n")
            print("Ready!\n")

            # State variables
            last_fire = 0.0         # Timestamp of last action
            last_note = None        # Last detected note name
            last_note_time = 0.0    # Timestamp of last note detection
            ema = noise_floor       # Adaptive noise floor (EMA)

            # Main detection loop
            while True:
                chunk = audio_q.get()
                level = rms(chunk)

                # Update adaptive noise floor when signal is quiet
                # This tracks slowly changing background noise
                if level < max(ONSET_ABS, ema * 1.4):
                    ema = 0.98 * ema + 0.02 * level

                # Onset detection: is this a fresh pluck?
                # Requires signal to be both:
                # 1. Above absolute threshold (ONSET_ABS)
                # 2. Significantly louder than noise floor (ONSET_RATIO)
                onset = (level > max(ONSET_ABS, ema * ONSET_RATIO))

                now = time.monotonic()
                
                # Check if onset detected AND global cooldown expired
                if onset and (now - last_fire) > COOLDOWN_SEC:
                    # Collect additional chunks for better pitch accuracy
                    chunks = [chunk]
                    for _ in range(POST_ONSET_CHUNKS - 1):
                        try:
                            chunks.append(audio_q.get(timeout=0.5))
                        except queue.Empty:
                            break  # Use what we have if queue is empty
                    
                    y = np.concatenate(chunks)
                    
                    # Detect pitch
                    freq, conf = detect_pitch_fft(y, SR)
                    
                    if debug:
                        print(f"[DEBUG] level={level:.6f} freq={freq:.1f}Hz conf={conf:.2f}")

                    # Check if confidence is high enough
                    if conf >= CONFIDENCE_MIN:
                        matched_note, error = match_note(freq, templates)
                        
                        if matched_note and matched_note in NOTE_TO_KEY:
                            # Same-note cooldown check
                            # Prevents double-triggers from string decay
                            time_since_same_note = (now - last_note_time 
                                                   if matched_note == last_note 
                                                   else 999)
                            
                            if time_since_same_note > SAME_NOTE_COOLDOWN:
                                # Trigger allowed!
                                key = NOTE_TO_KEY[matched_note]
                                keyboard.press(key)
                                keyboard.release(key)
                                
                                # Update state
                                last_fire = now
                                last_note = matched_note
                                last_note_time = now
                                
                                # Flush queue to prevent lag
                                flush_queue(audio_q)
                                
                                if debug:
                                    print(f"✓ {matched_note} ({freq:.1f}Hz, "
                                         f"Δ{error:.1f}Hz) → {key}\n")
                            else:
                                # Same note too soon - block to prevent double-trigger
                                if debug:
                                    print(f"✗ Same note too soon: {matched_note} "
                                         f"(waited {time_since_same_note:.2f}s "
                                         f"< {SAME_NOTE_COOLDOWN}s)\n")
                        else:
                            # No matching note found
                            if debug:
                                print(f"✗ No match: {freq:.1f}Hz "
                                     f"(error: {error:.1f}Hz)\n")
                    else:
                        # Confidence too low
                        if debug:
                            print(f"✗ Low conf: {conf:.2f}\n")
    
    except KeyboardInterrupt:
        print("\n[STOPPED]")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Parse arguments and run calibration or detection mode."""
    parser = argparse.ArgumentParser(
        description="ChordSurfer - Play games with your guitar!",
        epilog="Example: python main.py --calibrate"
    )
    parser.add_argument(
        "--list-devices", 
        action="store_true",
        help="List available audio input devices and exit"
    )
    parser.add_argument(
        "--device", 
        type=int, 
        default=None,
        help="Audio input device index (see --list-devices)"
    )
    parser.add_argument(
        "--calibrate", 
        action="store_true",
        help="Run calibration mode to learn your guitar's notes"
    )
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug output (shows detection details)"
    )
    parser.add_argument(
        "--templates", 
        type=str, 
        default=TEMPLATES_FILE,
        help=f"Path to templates file (default: {TEMPLATES_FILE})"
    )
    
    args = parser.parse_args()

    # List devices and exit
    if args.list_devices:
        list_devices()
        return

    device = int(args.device) if args.device is not None else None
    notes = tuple(NOTE_TO_KEY.keys())

    # Calibration mode
    if args.calibrate:
        calibrate(args.templates, device, notes)
        return

    # Check if templates exist
    if not os.path.exists(args.templates):
        print(f"[ERROR] Missing: {args.templates}")
        print("Run: python main.py --calibrate")
        return

    # Load templates
    try:
        templates = load_templates(args.templates)
    except (ValueError, json.JSONDecodeError) as e:
        print(f"[ERROR] Bad templates: {e}")
        print("Run: python main.py --calibrate")
        return

    # Run detection loop
    detect_loop(templates, device, debug=args.debug)


if __name__ == "__main__":
    main()