# PitchView

Real-time spectrum and note viewer for whatever your PC is playing. Play a piano song and see which notes are sounding, as they sound.

Windows 10 / 11, 64-bit.

## Run

Unzip the folder anywhere and double-click **PitchView.exe**. You don't need to install anything.

PitchView listens to your default speakers or headphones by default. To use a different output, or a microphone, open **Settings → Advanced → Input**. Press **↻** there after plugging in or removing a device.

## Screen

- **Spectrum**
  - Drag the two white walls to choose which frequencies are analysed. Everything outside them is hatched and ignored.
- **Level meter** (right of the spectrum)
  - The pink line is the threshold: quieter peaks are ignored.
  - The upper line is the loudness where colours are brightest.
  - Drag either line, or scroll over it.
- **Tilt** (top right)
  - Tilts the spectrum around 1 kHz.
  - A negative value helps quiet high notes reach the threshold.
  - Double-click to reset it.
- **Piano roll**
  - **Keys** shows detected notes per key.
  - **Freq** shows the spectrum scrolling up, Wave Candy style.

Hover over any setting's label to see what it does. **Settings → Advanced → Reset all settings** restores the defaults. Space pauses the display.

## Tips for piano

- Set the threshold just under the quiet notes. Leave the upper colour line around −10 dB.
- Keep **Harmonics → Hide** on, so only the played notes are shown and not their overtones.
- If note names are a bit off for a whole song, the recording is probably not tuned to A4 = 440 Hz. Adjust **Settings → Advanced → A4 reference**.

## Build from source

```
pip install -r requirements.txt pyinstaller
python build.py
```

This creates `dist/PitchView/PitchView.exe` and a zip in `release/`. To run without building, use `python pitchview.py`.
