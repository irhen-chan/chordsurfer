# ChordSurfer 🎸🎮

Play Subway Surfers with your guitar.

## What is this?

Turns your guitar into a game controller by detecting which notes you play and mapping them to arrow keys



## How it works

- Listens to your guitar through your computer's microphone
- Detects pitch using FFT analysis
- Maps specific notes to arrow key presses
- Sends those keypresses to whatever game window has focus

Current mapping:
- **G3** (3rd string open) → UP (jump)
- **B3** (2nd string open) → DOWN (roll)  
- **D4** (2nd string 3rd fret) → LEFT
- **E4** (1st string open) → RIGHT

## Why these specific notes?

Tried low bass notes first (E2, A2, etc.) but laptop mics are terrible at picking up low frequencies. Higher notes (196-329 Hz range) work way better for detection.

## Setup

**Requirements:**
- Python 3.8+
- A guitar
- Working microphone
- Windows

**Install:**
```bash
pip install -r requirements.txt
```

**Calibrate:**
```bash
python main.py --calibrate
```

Follow the prompts. You'll pluck each note for 2 seconds so the system learns your guitar's frequencies.

**Play:**
```bash
python main.py
```

Open Subway Surfers (or any game that uses arrow keys), click in the game window, and start playing.

## Tips

- Use headphones so game audio doesn't confuse the mic
- Pluck
- Make sure Windows input volume is at 80-100%
- Mic should be 20-30cm from guitar

## Troubleshooting

**Nothing is being detected:**
```bash
python mic_diagnostic.py
```
This shows if your mic is actually picking up the guitar.

**Getting double-triggers:**
The code has cooldown logic to prevent this, but you can tune it in `main.py`:
- `SAME_NOTE_COOLDOWN = 0.5` - how long before you can trigger the same note again
- `COOLDOWN_SEC = 0.35` - minimum time between any actions

## Files

- `main.py` - The actual program
- `mic_diagnostic.py` - Test if your mic is working
- `pitch_diagnostic.py` - Debug pitch detection issues
- `test_keys.py` - Verify keyboard simulation works
- `note_templates.json` - Your calibrated note frequencies (created during calibration)


## License

MIT 

---
