"""PitchView - real-time spectrum + pitch viewer for whatever the PC is playing.

Captures system output via WASAPI loopback (or a microphone), runs an FFT,
keeps only the strongest peaks above a threshold and names them as notes.
Views: spectrum with a level meter (where the threshold is set), 88-key piano roll.
"""
import os
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np
import pyaudiowpatch as pa
from PyQt5 import QtCore, QtGui, QtWidgets

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
PIANO_LO, PIANO_HI = 21, 108  # A0 .. C8
MIN_RATIO = 2 ** (2 / 12)           # walls stay at least a whole tone apart


def freq_to_midi(f, a4):
    return 69.0 + 12.0 * np.log2(f / a4)


def midi_to_freq(m, a4):
    return a4 * 2.0 ** ((m - 69) / 12.0)


def midi_name(m):
    n = int(round(m))
    return f"{NOTE_NAMES[n % 12]}{n // 12 - 1}"


def is_black(m):
    return (m % 12) in (1, 3, 6, 8, 10)


WALL_INSET = 0.03  # default walls sit this fraction of the view in from each edge


def default_walls(view_lo, view_hi):
    """Wall frequencies WALL_INSET in from both ends of the (log) view."""
    x0, x1 = np.log10(view_lo), np.log10(view_hi)
    span = x1 - x0
    return float(10 ** (x0 + WALL_INSET * span)), float(10 ** (x1 - WALL_INSET * span))


def piano_view_range(a4):
    """Frequency span whose log axis lines up with the 88-key keyboard drawn edge to edge.

    On the keyboard one octave is always 7 white keys wide and each C sits at the
    centre of its white key, so a log-frequency axis matches it exactly at every C
    when its ends are placed at the keyboard's outer edges.
    """
    whites = [m for m in range(PIANO_LO, PIANO_HI + 1) if not is_black(m)]
    c4 = whites.index(60) + 0.5          # C4's centre, in white-key widths from the left edge
    semis_per_white = 12 / 7
    lo = 60 - c4 * semis_per_white
    hi = 60 + (len(whites) - c4) * semis_per_white
    return float(midi_to_freq(lo, a4)), float(midi_to_freq(hi, a4))


def blackman_harris(n):
    k = np.arange(n) / (n - 1)
    return (0.35875 - 0.48829 * np.cos(2 * np.pi * k)
            + 0.14128 * np.cos(4 * np.pi * k) - 0.01168 * np.cos(6 * np.pi * k))


# --------------------------------------------------------------------------- themes

THEMES = {
    # window, panels and controls; the loudness colours come from GRADIENTS
    "Dark": dict(  # charcoal, after Xynth Audio Chroma
        bg='#1c1c1f',
        panel='#121214',
        fg='#e6e3ec',
        dim='#8b8896',
        button='#2b2b30',
        accent='#b48cf0',
        accent2='#5aa8f0',
        curve='#b48cf0',
        ghost=(180, 140, 240, 45),
        thr='#f07fb8',
        lane_w='#151517',
        lane_b='#101012',
        guide='#28272e',
        key_w='#34343a',
        key_b='#0f0f11'),
    "White": dict(  # light
        bg='#efebe4',
        panel='#faf8f4',
        fg='#2b2723',
        dim='#8c8378',
        button='#e3ddd3',
        accent='#6a48c8',
        accent2='#2f7fd0',
        curve='#5a4fb0',
        ghost=(90, 79, 176, 50),
        thr='#d0453a',
        lane_w='#f6f3ee',
        lane_b='#ebe6de',
        guide='#d6cdbf',
        key_w='#ffffff',
        key_b='#2f2b27'),
}


# colour-map presets, dark (quiet) to light (loud); any of them can replace the theme's own
GRADIENTS = {
    "Glacier": ["#1b1f4b", "#3a5bd9", "#4fc3f7", "#b8f2e6", "#ffffff"],
    "Twilight": [(0, '#2a2350'), (0.3, '#6a48c8'), (0.55, '#4f9fec'), (0.8, '#b48cf0'), (1, '#f5efff')],
    "Aurora": [(0, '#123a4a'), (0.3, '#1fa6a0'), (0.6, '#8f7cf0'), (0.85, '#ff8fd0'), (1, '#fff0fa')],
    "Ember": [(0, '#3d1414'), (0.3, '#a3262a'), (0.6, '#f0652f'), (0.85, '#ffc061'), (1, '#fff6dc')],
    "Paper": [(0, '#d5ccf2'), (0.35, '#8f7ae0'), (0.65, '#4f7fd8'), (1, '#1b2a78')],
    # each preset passes through several hues while still rising from dark (quiet) to light (loud)
    "Magma": ["#2a0b4a", "#8c2981", "#de4968", "#fe9f6d", "#fcfdbf"],
    "Viridis": ["#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"],
    "Ocean": ["#141b4d", "#1f5f8b", "#16b3a6", "#9be7a8", "#f4fff2"],
    "Sunset": ["#2d1b4e", "#7b2d8e", "#e0457b", "#ffa45b", "#fff1c1"],
    "Forest": ["#10243a", "#1b6b4a", "#6cbf43", "#e6d94c", "#fff8d6"],
    "Fire": ["#1e0b2c", "#8b1e3f", "#e2501c", "#ffb02e", "#fff3b0"],
    "Neon": ["#1a0b3d", "#5b1fd6", "#00c2ff", "#39ffb0", "#f2ffde"],
    "Blossom": ["#2b1033", "#8e2c6f", "#e85d9a", "#ffb38a", "#fff2e6"],
    "Pastel": ["#2d2a4a", "#6a5acd", "#8ecae6", "#ffc8dd", "#fffbe6"],
}
GRADIENT_CHOICES = list(GRADIENTS) + ["Random"]


def random_gradient(rng=None):
    """Five stops that rise in lightness while drifting through a random stretch of hues."""
    import colorsys
    rng = rng or np.random.default_rng()
    h0 = rng.random()
    span = rng.uniform(0.15, 0.45) * rng.choice([-1, 1])
    sat = rng.uniform(0.55, 0.9)
    stops = []
    for i, light in enumerate((0.18, 0.35, 0.55, 0.75, 0.93)):
        r, g, b = colorsys.hls_to_rgb((h0 + span * i / 4) % 1.0, light, sat * (1 - 0.35 * (i == 4)))
        stops.append("#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255)))
    return stops


def build_lut(stops):
    """256-entry colour map -> (uint32 ARGB array, list of QColor, float RGB array)."""
    pos = np.array([s[0] for s in stops])
    cols = np.array([QtGui.QColor(s[1]).getRgb()[:3] for s in stops], float)
    t = np.linspace(0, 1, 256)
    rgb = np.stack([np.interp(t, pos, cols[:, i]) for i in range(3)], 1).astype(np.uint32)
    argb = (0xFF000000 | (rgb[:, 0] << 16) | (rgb[:, 1] << 8) | rgb[:, 2]).astype(np.uint32)
    return argb, [QtGui.QColor(int(r), int(g), int(b)) for r, g, b in rgb], rgb.astype(np.float32)


def level_gradient(lut_q, r, h_thr, h_top):
    """Loudness colour map laid out vertically over rect r: dim below the threshold height,
    full colour at the colour-max height (heights are 0..1 from the bottom)."""
    g = QtGui.QLinearGradient(0, r.bottom(), 0, r.top())
    low = QtGui.QColor(lut_q[0])
    low.setAlpha(70)
    g.setColorAt(0, low)
    g.setColorAt(h_thr, low)
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        g.setColorAt(h_thr + t * (h_top - h_thr), lut_q[int(t * 255)])
    g.setColorAt(1, lut_q[255])
    return QtGui.QBrush(g)


VERSION = "1.1.0"
FROZEN = getattr(sys, "frozen", False)  # running as the packaged .exe
# bundled files (fonts, icon) live next to the script, or in PyInstaller's unpack folder when frozen
APP_DIR = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
FONT_DIR = os.path.join(APP_DIR, "fonts")
ICON_PATH = os.path.join(APP_DIR, "assets", "pitchview.ico")
UI_FONT = "Gowun Dodum"  # bundled in fonts/ (SIL Open Font License); falls back to the system font if missing


def load_fonts():
    """Register the bundled UI font with Qt and make it the application default."""
    if not os.path.isdir(FONT_DIR):
        return
    loaded = False
    for name in sorted(os.listdir(FONT_DIR)):
        if name.lower().endswith((".otf", ".ttf")):
            loaded |= QtGui.QFontDatabase.addApplicationFont(os.path.join(FONT_DIR, name)) != -1
    if loaded:
        font = QtGui.QFont(UI_FONT)
        font.setPixelSize(13)  # pixels, not points: the same size on every display scale
        font.setHintingPreference(QtGui.QFont.PreferNoHinting)  # keeps the font's rounded shapes intact on Windows
        QtWidgets.QApplication.instance().setFont(font)


def resample_columns(px, values, n):
    """Spread per-bin values (at pixel positions px, ascending) over n pixel columns:
    the max where bins are dense, linear interpolation where they are sparse."""
    col = np.zeros(n, np.float32)
    if len(px) > 1:
        ok = (px >= 0) & (px < n)
        np.maximum.at(col, px[ok].astype(np.int32), values[ok])
        col = np.maximum(col, np.interp(np.arange(n) + 0.5, px, values, left=0, right=0))
    return col


def readable(color, dark_bg):
    """The same hue, made light enough for a dark background (or dark enough for a light one)."""
    h, s, l, _ = color.getHslF()
    l = max(l, 0.62) if dark_bg else min(l, 0.42)
    return QtGui.QColor.fromHslF(max(h, 0.0), s, l)


def argb(hex_color):
    c = QtGui.QColor(hex_color)
    return np.uint32(0xFF000000 | (c.red() << 16) | (c.green() << 8) | c.blue())


# --------------------------------------------------------------------------- audio

class AudioCapture:
    """Mono ring buffer fed by a PyAudio callback stream."""

    BUF_LEN = 1 << 16

    def __init__(self, pyaudio, device):
        self.p = pyaudio
        self.device = device
        self.rate = int(device["defaultSampleRate"])
        self.channels = max(1, min(int(device["maxInputChannels"]), 8))
        self.buf = np.zeros(self.BUF_LEN, dtype=np.float32)
        self.lock = threading.Lock()
        self.last_data = time.perf_counter()
        self.stream = None
        self.heard = False

    def _callback(self, in_data, frame_count, time_info, status):
        x = np.frombuffer(in_data, dtype=np.float32)
        if self.channels > 1:
            x = x.reshape(-1, self.channels).mean(axis=1)
        self._push(x)
        if not self.heard and np.any(x):
            self.heard = True  # at least one non-zero sample arrived
        self.last_data = time.perf_counter()
        return None, pa.paContinue

    def _push(self, x):
        n = len(x)
        if n == 0:
            return
        with self.lock:
            if n >= self.BUF_LEN:
                self.buf[:] = x[-self.BUF_LEN:]
            else:
                self.buf[:-n] = self.buf[n:]
                self.buf[-n:] = x

    def start(self):
        self.stream = self.p.open(
            format=pa.paFloat32, channels=self.channels, rate=self.rate,
            input=True, input_device_index=self.device["index"],
            frames_per_buffer=512, stream_callback=self._callback)
        self.stream.start_stream()

    def stop(self):
        if self.stream is not None:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

    def pad_silence(self):
        # WASAPI loopback delivers nothing while the system is silent,
        # so feed zeros to let the display decay instead of freezing.
        now = time.perf_counter()
        gap = now - self.last_data
        if gap > 0.08:
            self._push(np.zeros(min(int(gap * self.rate), self.BUF_LEN), np.float32))
            self.last_data = now

    def latest(self, n):
        with self.lock:
            return self.buf[-n:].copy()


# --------------------------------------------------------------------------- analysis

class Analyzer:
    PAD = 2  # zero-padding factor for finer peak interpolation

    def __init__(self):
        self.n = 0
        self.rate = 0
        self.smoothed = None
        self.set_size(8192, 48000)

    def set_size(self, n, rate):
        if n == self.n and rate == self.rate:
            return
        self.n, self.rate = n, rate
        self.window = blackman_harris(n).astype(np.float32)
        self.norm = 2.0 / self.window.sum()
        self.freqs = np.fft.rfftfreq(n * self.PAD, 1.0 / rate)
        self.smoothed = None

    def spectrum(self, x, smoothing):
        mag = np.abs(np.fft.rfft(x * self.window, n=self.n * self.PAD)) * self.norm
        db = 20.0 * np.log10(mag + 1e-9)
        if self.smoothed is None or smoothing <= 0:
            self.smoothed = db
        else:
            # fast attack, smoothed release
            rel = smoothing * self.smoothed + (1 - smoothing) * db
            self.smoothed = np.maximum(db, rel)
        return self.smoothed

    def peaks(self, db, lo_idx, hi_idx, threshold, rel_range, max_peaks, a4, drop_harmonics, tol_cents=50):
        s = db[lo_idx:hi_idx]
        if len(s) < 3:
            return []
        mid = s[1:-1]
        cand = np.nonzero((mid > s[:-2]) & (mid >= s[2:]) & (mid > threshold))[0] + 1
        if len(cand) == 0:
            return []
        a, b, c = s[cand - 1], s[cand], s[cand + 1]
        denom = a - 2 * b + c
        p = np.where(denom != 0, 0.5 * (a - c) / np.where(denom == 0, 1, denom), 0.0)
        peak_db = b - 0.25 * (a - c) * p
        binhz = self.freqs[1]
        peak_f = (lo_idx + cand + p) * binhz

        # keep only peaks close to a piano key; noise, percussion and inharmonic
        # partials land between keys. 50 cents (half a semitone) accepts everything.
        ok = peak_f > 0
        peak_m = np.zeros_like(peak_f)
        peak_m[ok] = freq_to_midi(peak_f[ok], a4)
        if tol_cents < 50:
            ok &= np.abs(peak_m - np.round(peak_m)) * 100 <= tol_cents
        if not ok.any():
            return []
        peak_f, peak_db, peak_m = peak_f[ok], peak_db[ok], peak_m[ok]

        order = np.argsort(peak_db)[::-1]
        top = peak_db[order[0]]
        chosen = []  # (freq, db, midi)
        for i in order:
            if rel_range > 0 and peak_db[i] < top - rel_range:
                break
            f, m = peak_f[i], peak_m[i]
            # merge peaks closer than ~half a semitone into the stronger one
            if any(abs(m - cm) < 0.5 for _, _, cm in chosen):
                continue
            chosen.append((f, float(peak_db[i]), m))
            if len(chosen) >= max_peaks * 3:
                break

        if drop_harmonics and len(chosen) > 1:
            chosen.sort(key=lambda t: t[0])
            keep = []
            for f, d, m in chosen:
                harmonic = False
                for g, gd, _ in keep:
                    k = round(f / g)
                    if k >= 2 and abs(12 * np.log2(f / (k * g))) < 0.35 and d < gd + 6:
                        harmonic = True
                        break
                if not harmonic:
                    keep.append((f, d, m))
            chosen = keep

        chosen.sort(key=lambda t: t[1], reverse=True)
        return chosen[:max_peaks]


# --------------------------------------------------------------------------- widgets

class PianoRoll(QtWidgets.QWidget):
    """Scrolling history over an 88-key keyboard (newest row at the bottom).

    Two views of the same timeline:
      keys - detected notes, one lane per key
      freq - the spectrum itself, Wave Candy style, on the same log axis as the spectrum view
    levels / freq rows: -1 = nothing, 0..1 = loudness position on the colour map.
    """

    KB_H = 70
    BASE_SPEED = 4.0  # rows per 1/60 s at 1x (240 px/s)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(160)
        self.levels = np.full(PIANO_HI - PIANO_LO + 1, -1, np.float32)
        self.hist = None    # note history, float32 (h, 88)
        self.hist_f = None  # spectrum history, float32 (h, width)
        self.hist_rgb = None  # hist_f already blended to colours; only new rows are coloured each frame
        self.mode = "keys"
        self.speed = 1.0  # user-facing multiplier; 1x scrolls BASE_SPEED rows per 1/60 s
        self.frame_scale = 1.0  # elapsed time of the current frame, in 1/60 s units (set by the main loop)
        self._acc = 0.0
        self.paused = False
        self._geom_w = -1

    def set_theme(self, th, lut, lut_q, lut_rgb, *_):
        self.th, self.lut, self.lut_q, self.lut_rgb = th, lut, lut_q, lut_rgb
        self.hist_rgb = None
        self._geom_w = -1
        self.update()

    def _layout(self, w):
        n_white = sum(1 for m in range(PIANO_LO, PIANO_HI + 1) if not is_black(m))
        ww = w / n_white
        self.keys = []  # (midi, x0, x1, black)
        wi = 0
        for m in range(PIANO_LO, PIANO_HI + 1):
            if is_black(m):
                cx = wi * ww
                self.keys.append((m, cx - 0.3 * ww, cx + 0.3 * ww, True))
            else:
                self.keys.append((m, wi * ww, (wi + 1) * ww, False))
                wi += 1
        keymap = np.zeros(w, np.int32)
        base = np.zeros(w, np.uint32)
        lw, lb = argb(self.th["lane_w"]), argb(self.th["lane_b"])
        for black in (False, True):
            for m, x0, x1, b in self.keys:
                if b != black:
                    continue
                a, z = int(round(x0)), max(int(round(x0)) + 1, int(round(x1)))
                keymap[a:z] = m - PIANO_LO
                base[a:z] = lb if b else lw
        guide = argb(self.th["guide"])
        for m, x0, _, b in self.keys:  # faint guide line on every C
            if m % 12 == 0:
                base[min(int(round(x0)), w - 1)] = guide
        # lanes are background: premultiplied with the window's background opacity
        alpha = self.th.get("alpha", 1.0)
        self.base_rgb = np.stack([(base >> 16) & 255, (base >> 8) & 255, base & 255], 1).astype(np.float32)
        pm = (self.base_rgb * alpha).astype(np.uint32)
        base = (np.uint32(int(alpha * 255)) << 24) | (pm[:, 0] << 16) | (pm[:, 1] << 8) | pm[:, 2]
        self.keymap, self.base = keymap, base.astype(np.uint32)
        self._geom_w = w

    def set_mode(self, mode):
        self.mode = mode
        self.update()

    def _colorize(self, lv):
        """Freq-roll rows -> premultiplied RGBA over the (possibly see-through) lanes: quiet = almost background."""
        a = np.clip(lv * 1.4, 0, 1)[..., None]
        bg_a = self.th.get("alpha", 1.0)
        col = self.lut_rgb[(np.clip(lv, 0, 1) * 255).astype(np.int32)]
        rgb = col * a + self.base_rgb * bg_a * (1 - a)
        alpha = (a + bg_a * (1 - a)) * 255
        return np.concatenate([rgb, alpha], axis=-1).astype(np.uint8)

    def _rgb_valid(self):
        return (self.hist_rgb is not None and self.hist_f is not None
                and self.hist_rgb.shape[:2] == self.hist_f.shape
                and getattr(self, "base_rgb", None) is not None and len(self.base_rgb) == self.hist_f.shape[1])

    @staticmethod
    def _scroll(hist, rows, row):
        if rows == 0:  # slower than one row per frame: merge into the newest row
            np.maximum(hist[-1], row, out=hist[-1])
        else:
            hist[:-rows] = hist[rows:]
            hist[-rows:] = row

    def push(self, levels, backfill=(), freq_row=None):
        self.levels = levels
        h = max(1, self.height() - self.KB_H)
        if self.hist is None or self.hist.shape[0] != h:
            self.hist = np.full((h, len(levels)), -1, np.float32)
        if freq_row is not None and (self.hist_f is None or self.hist_f.shape != (h, len(freq_row))):
            self.hist_f = np.full((h, len(freq_row)), -1, np.float32)
        if not self.paused:
            self._acc += self.speed * self.BASE_SPEED * self.frame_scale
            rows = min(int(self._acc), h)
            self._acc -= int(self._acc)
            self._scroll(self.hist, rows, levels)
            if freq_row is not None:
                self._scroll(self.hist_f, rows, freq_row)
                if self._rgb_valid():
                    n = max(rows, 1)
                    if rows:
                        self.hist_rgb[:-rows] = self.hist_rgb[rows:]
                    self.hist_rgb[-n:] = self._colorize(self.hist_f[-n:])
            for k, n in backfill:  # draw a newly confirmed note from its actual onset
                back = int(round(n * self.speed * self.BASE_SPEED * self.frame_scale))
                if back > 0:
                    seg = self.hist[-max(rows, 1) - back:-max(rows, 1), k]
                    np.maximum(seg, levels[k], out=seg)
        self.update()

    @staticmethod
    def _key_path(rect, radius, round_top):
        path = QtGui.QPainterPath()
        if round_top:
            path.addRoundedRect(rect, radius, radius)
            return path
        # square top, rounded bottom corners, traced directly (a path union of a rounded rect
        # and a square could drop the top half of a key)
        x0, y0, x1, y1, d = rect.left(), rect.top(), rect.right(), rect.bottom(), 2 * radius
        path.moveTo(x0, y0)
        path.lineTo(x1, y0)
        path.lineTo(x1, y1 - radius)
        path.arcTo(QtCore.QRectF(x1 - d, y1 - d, d, d), 0, -90)
        path.lineTo(x0 + radius, y1)
        path.arcTo(QtCore.QRectF(x0, y1 - d, d, d), 270, -90)
        path.closeSubpath()
        return path

    def _key_paths(self, y0):
        """Key outlines, rebuilt only when the size changes: white keys rounded at both ends,
        black keys rounded at the bottom only."""
        key = (self._geom_w, y0)
        if getattr(self, "_paths_key", None) != key:
            paths = {}
            for m, x0, x1, b in self.keys:
                if b:
                    rect = QtCore.QRectF(x0, y0, x1 - x0, self.KB_H * 0.62)
                    paths[m] = self._key_path(rect, min(3.0, rect.width() / 3), False)
                else:
                    rect = QtCore.QRectF(x0 + 0.75, y0 + 1, x1 - x0 - 1.5, self.KB_H - 2)
                    paths[m] = self._key_path(rect, min(4.0, rect.width() / 4), True)
            self._paths, self._paths_key = paths, key
        return self._paths

    def paintEvent(self, ev):
        qp = QtGui.QPainter(self)
        w, h = self.width(), self.height()
        if self._geom_w != w:
            self._layout(w)
        hh = h - self.KB_H
        drawn = False
        if self.mode == "freq" and self.hist_f is not None and self.hist_f.shape == (hh, w):
            if not self._rgb_valid():  # after a resize or theme change: colour the whole history once
                self.hist_rgb = self._colorize(self.hist_f)
            img = np.ascontiguousarray(self.hist_rgb)
            qp.drawImage(0, 0, QtGui.QImage(img.data, w, hh, w * 4, QtGui.QImage.Format_RGBA8888_Premultiplied))
            lv = None
            drawn = True
        elif self.hist is not None and self.hist.shape[0] == hh:
            lv = self.hist[:, self.keymap]
        else:
            lv = None
        if lv is not None:
            img = np.where(lv >= 0, self.lut[(np.clip(lv, 0, 1) * 255).astype(np.int32)], self.base)
            img = np.ascontiguousarray(img, np.uint32)
            qimg = QtGui.QImage(img.data, w, hh, w * 4, QtGui.QImage.Format_ARGB32_Premultiplied)
            qp.drawImage(0, 0, qimg)
        elif not drawn:
            lane = QtGui.QColor(self.th["lane_w"])
            lane.setAlphaF(self.th.get("alpha", 1.0))
            qp.fillRect(0, 0, w, hh, lane)

        y0 = hh
        qp.setRenderHint(QtGui.QPainter.Antialiasing, True)
        font = qp.font()
        font.setPointSize(7)
        qp.setFont(font)
        kw, kb = QtGui.QColor(self.th["key_w"]), QtGui.QColor(self.th["key_b"])
        paths = self._key_paths(y0)
        for black in (False, True):
            for m, x0, x1, b in self.keys:
                if b != black:
                    continue
                lv = self.levels[m - PIANO_LO]
                path = paths[m]
                qp.fillPath(path, self.lut_q[int(np.clip(lv, 0, 1) * 255)] if lv >= 0 else (kb if b else kw))
                if b:  # thin outline so black keys stay separate from lit neighbours
                    qp.setPen(QtGui.QPen(QtGui.QColor(self.th["panel"]), 1))
                    qp.setBrush(QtCore.Qt.NoBrush)
                    qp.drawPath(path)
                elif m % 12 == 0:
                    qp.setPen(QtGui.QColor(self.th["dim"]))
                    qp.drawText(path.boundingRect().adjusted(0, 0, 0, -4),
                                QtCore.Qt.AlignHCenter | QtCore.Qt.AlignBottom, midi_name(m))
        qp.end()


class DragStep(QtCore.QObject):
    """Press on a number field and drag up/down to change it; a plain click selects the text for typing."""

    PX_PER_STEP = 4

    def __init__(self, spin):
        super().__init__(spin)
        self.spin = spin
        self.start = None
        self.dragged = False
        edit = spin.lineEdit()
        edit.setCursor(QtCore.Qt.SizeVerCursor)
        edit.installEventFilter(self)

    def eventFilter(self, obj, ev):
        t = ev.type()
        if t == QtCore.QEvent.MouseButtonPress and ev.button() == QtCore.Qt.LeftButton:
            self.start = (ev.globalPos().y(), self.spin.value())
            self.dragged = False
            return True
        if t == QtCore.QEvent.MouseMove and self.start is not None:
            dy = self.start[0] - ev.globalPos().y()  # up = bigger
            if abs(dy) >= 3:
                self.dragged = True
            if self.dragged:
                self.spin.setValue(self.start[1] + int(dy / self.PX_PER_STEP) * self.spin.singleStep())
            return True
        if t == QtCore.QEvent.MouseButtonRelease and self.start is not None:
            self.start = None
            if not self.dragged:
                obj.setFocus()
                obj.selectAll()
            return True
        return False


class NoSelectStep(QtCore.QObject):
    """Clicking the in-box arrows makes Qt select the whole number; drop that highlight right after."""

    def eventFilter(self, obj, ev):
        if ev.type() in (QtCore.QEvent.MouseButtonPress, QtCore.QEvent.MouseButtonRelease):
            QtCore.QTimer.singleShot(0, obj.lineEdit().deselect)
        return False


def stepper(spin, compact=False):
    """Number field with up/down arrows inside it (hold to repeat); the field can also be dragged."""
    spin.setButtonSymbols(QtWidgets.QAbstractSpinBox.UpDownArrows)
    spin.setAlignment(QtCore.Qt.AlignCenter)
    if compact:
        spin.setObjectName("compact")
    DragStep(spin)
    spin._no_select = NoSelectStep(spin)
    spin.installEventFilter(spin._no_select)
    return spin


class GlowSpectrum(QtWidgets.QWidget):
    """The spectrum view used by every spectrum mode.

    All modes share one frame: a dB axis on the left, note names along the
    bottom, draggable walls and the threshold line. Heights map dB linearly
    from `floor` (bottom edge) to 0 dBFS (top edge).

    Modes (UI name in brackets):
      both  [Prism] - glow lines clipped to the area under the spectrum
      glow  [Beam]  - glow lines only (after Xynth Chroma)
      curve [Line]  - lines only: dim full curve, bright curve above the threshold, peak dots
    """

    LABEL_H = 18     # bottom strip for note names
    AXIS_W = 34      # left strip for dB labels
    RADIUS = 12
    GRAB = 7         # px either side of a wall / the threshold line that starts a drag

    wallMoved = QtCore.pyqtSignal(int, float)   # wall index (0 = low, 1 = high), x in 0..1

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(160)
        self.setMouseTracking(True)
        self.walls = [0.0, 1.0]
        self._drag = None       # index of the wall being dragged
        self.mode = "both"
        self.floor, self.thr, self.top = -90.0, -60.0, -12.0
        self.xpos = np.zeros(0)
        self.lv = np.zeros(0)
        self.shape = None
        self.hcol = None
        self.peaks = []   # (xnorm, name, level, height)
        self.guides = []  # (xnorm, label)

    def set_theme(self, th, lut, lut_q, lut_rgb):
        self.th, self.lut_q, self.lut_rgb = th, lut_q, lut_rgb
        c = QtGui.QColor(th["panel"])
        self.bg_rgb = np.array([c.red(), c.green(), c.blue()], np.float32)
        self.update()

    def set_walls(self, lo, hi):
        self.walls = [lo, hi]
        self.update()

    def set_mode(self, mode):
        self.mode = mode
        self.update()

    def set_levels(self, floor, thr, top):
        self.floor, self.thr, self.top = floor, thr, top
        self.update()

    def set_data(self, xpos, lv, shape, peaks, guides):
        self.xpos, self.lv, self.shape, self.peaks, self.guides = xpos, lv, shape, peaks, guides
        self.update()

    # ---- geometry -----------------------------------------------------------
    def _plot_rect(self):
        return QtCore.QRectF(self.AXIS_W, 0, max(1, self.width() - self.AXIS_W),
                             max(1, self.height() - self.LABEL_H))

    def _height(self, db):
        return float(np.clip((db - self.floor) / -self.floor, 0, 1))

    def _y_of_db(self, db):
        r = self._plot_rect()
        return r.bottom() - self._height(db) * r.height()

    def _x_of(self, xn):
        r = self._plot_rect()
        return r.left() + xn * r.width()

    def _hit(self, pos):
        if not self._plot_rect().contains(pos):
            return None
        dist = [abs(pos.x() - self._x_of(self.walls[i])) for i in (0, 1)]
        i = int(np.argmin(dist))
        return i if dist[i] <= self.GRAB else None

    # ---- mouse --------------------------------------------------------------
    def mousePressEvent(self, ev):
        if ev.button() == QtCore.Qt.LeftButton:
            self._drag = self._hit(QtCore.QPointF(ev.pos()))

    def mouseMoveEvent(self, ev):
        pos = QtCore.QPointF(ev.pos())
        r = self._plot_rect()
        if self._drag is not None:
            self.wallMoved.emit(self._drag, float(np.clip((pos.x() - r.left()) / r.width(), 0, 1)))
            return
        if self._hit(pos) is not None:
            self.setCursor(QtCore.Qt.SizeHorCursor)
        else:
            self.unsetCursor()

    def mouseReleaseEvent(self, ev):
        self._drag = None

    # ---- rendering ----------------------------------------------------------
    def _columns(self, n, values):
        return resample_columns(self.xpos * n, values, n)

    def _render(self, w, h):
        masked = self.mode == "both"
        col = self._columns(w, self.lv) ** 1.8  # sharpen: keep peaks, suppress the window's shoulders
        k = np.exp(-0.5 * (np.arange(-15, 16) / 5.0) ** 2).astype(np.float32)
        glow = np.convolve(col, k / k.sum(), "same")
        inten = np.clip(np.maximum(col, glow * 0.9), 0, 1)
        cidx = (np.clip(np.maximum(col, glow), 0, 1) ** 0.6 * 255).astype(np.int32)

        yy = np.linspace(-1, 1, h, dtype=np.float32)[:, None]
        yy = np.maximum(-yy, 0) if masked else np.abs(yy)
        half = 0.45 + 0.6 * np.sqrt(inten)[None, :]
        v = np.clip((half - yy) / 0.5, 0, 1) ** 1.5
        v *= np.where(inten > 0.002, 0.15 + 0.85 * np.sqrt(inten), 0)[None, :]
        if masked:
            # height above the bottom of each pixel row; 1px soft edge along the curve
            above = (h - 0.5 - np.arange(h, dtype=np.float32))[:, None]
            v *= np.clip(self.hcol[None, :] * h - above + 0.5, 0, 1)
        # premultiplied ARGB: colour v * lut with alpha v, so the card behind (opaque or see-through) shows
        # wherever there is no glow; over an opaque card this looks exactly like blending with its colour
        col = self.lut_rgb[cidx][None, :, :] * v[:, :, None]
        a = (v * 255).astype(np.uint32)
        c = col.astype(np.uint32)
        img = (a << 24) | (c[..., 0] << 16) | (c[..., 1] << 8) | c[..., 2]
        return np.ascontiguousarray(img, np.uint32)

    @staticmethod
    def _polyline(x0, ys):
        # drop points on flat stretches: antialiased stroking cost scales with the point count
        dy = np.abs(np.diff(ys)) > 0.25
        keep = np.zeros(len(ys), bool)
        keep[[0, -1]] = True
        keep[1:] |= dy
        keep[:-1] |= dy
        return [QtCore.QPointF(x0 + x + 0.5, float(ys[x])) for x in np.nonzero(keep)[0]]

    def _paint_axis(self, qp, r):
        th = self.th
        span = -self.floor
        step = 10 if span > 30 else 5
        font = qp.font()
        font.setPointSize(8)
        qp.setFont(font)
        for db in np.arange(0, self.floor - 1e-6, -step):  # labels only: no grid lines inside the plot
            y = self._y_of_db(db)
            ty = float(np.clip(y - 7, 0, r.bottom() - 14))
            qp.setPen(QtGui.QColor(th["dim"]))
            qp.drawText(QtCore.QRectF(0, ty, self.AXIS_W - 6, 14),
                        QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter, f"{db:.0f}")
        qp.drawText(QtCore.QRectF(0, r.bottom(), self.AXIS_W - 6, self.LABEL_H),
                    QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter, "dB")

    def _paint_walls(self, qp, r):
        """Everything outside the walls is dimmed and hatched: it is not analysed."""
        th = self.th
        shade = QtGui.QColor(th["panel"])
        shade.setAlphaF(0.88 * th.get("alpha", 1.0))  # dims like the rest of the background
        hatch = QtGui.QColor(th["dim"])
        hatch.setAlpha(45)
        for i in (0, 1):
            wx = self._x_of(self.walls[i])
            if i == 0:
                outside = QtCore.QRectF(r.left(), r.top(), wx - r.left(), r.height())
            else:
                outside = QtCore.QRectF(wx, r.top(), r.right() - wx, r.height())
            qp.fillRect(outside, shade)
            qp.fillRect(outside, QtGui.QBrush(hatch, QtCore.Qt.BDiagPattern))
            qp.setPen(QtGui.QPen(QtGui.QColor(th["fg"]), 1.6))
            qp.drawLine(QtCore.QPointF(wx, r.top()), QtCore.QPointF(wx, r.bottom()))

    def paintEvent(self, ev):
        qp = QtGui.QPainter(self)
        qp.setRenderHint(QtGui.QPainter.Antialiasing, True)
        w, h = self.width(), self.height()
        th = self.th
        path = QtGui.QPainterPath()
        path.addRoundedRect(QtCore.QRectF(0, 0, w, h), self.RADIUS, self.RADIUS)
        qp.setClipPath(path)  # no fill of its own: the card behind is the background

        r = self._plot_rect()
        x0, pw, ih = r.left(), int(r.width()), int(r.height())
        self._paint_axis(qp, r)

        qp.save()
        qp.setClipRect(r, QtCore.Qt.IntersectClip)
        self.hcol = self._columns(pw, self.shape) if self.shape is not None else None
        if self.mode in ("both", "glow"):
            # rendered at half height and scaled up: the vertical profile is smooth, so this halves the cost
            rh = max(1, ih // 2)
            img = self._render(pw, rh)
            qp.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, True)
            qp.drawImage(QtCore.QRectF(x0, 0, pw, ih),
                         QtGui.QImage(img.data, pw, rh, pw * 4, QtGui.QImage.Format_ARGB32_Premultiplied))
        if self.hcol is not None and self.mode != "glow":
            ys = ih - self.hcol * ih
            ridge = QtGui.QColor(th["curve"])
            if self.mode == "curve":  # lines only: dim full curve, bright curve above the threshold
                ghost = QtGui.QColor(*th["ghost"])
                qp.setPen(QtGui.QPen(ghost, 1.0))
                qp.setBrush(QtCore.Qt.NoBrush)
                qp.drawPolyline(QtGui.QPolygonF(self._polyline(x0, ys)))
                pts = self._polyline(x0, np.minimum(ys, self._y_of_db(self.thr)))
                ridge.setAlpha(230)
            else:
                pts = self._polyline(x0, ys)
                ridge.setAlpha(150)
            qp.setPen(QtGui.QPen(ridge, 1.0))  # 1px keeps Qt on its fast stroking path (>1px is ~50x slower)
            qp.setBrush(QtCore.Qt.NoBrush)
            qp.drawPolyline(QtGui.QPolygonF(pts))
        self._paint_walls(qp, r)
        if self.mode == "curve":  # threshold guide only in Line view; the level meter shows it everywhere
            y_thr = self._y_of_db(self.thr)
            qp.setPen(QtGui.QPen(QtGui.QColor(th["thr"]), 1.0, QtCore.Qt.DashLine))
            qp.drawLine(QtCore.QPointF(r.left(), y_thr), QtCore.QPointF(r.right(), y_thr))
        qp.restore()

        font = qp.font()
        font.setPointSize(8)
        qp.setFont(font)
        guide = QtGui.QColor(th["guide"])
        for xn, label in self.guides:
            x = self._x_of(xn)
            qp.setPen(QtGui.QPen(guide, 1))
            qp.drawLine(QtCore.QPointF(x, ih - 6), QtCore.QPointF(x, ih))
            qp.setPen(QtGui.QColor(th["dim"]))
            qp.drawText(QtCore.QRectF(x - 20, ih, 40, self.LABEL_H), QtCore.Qt.AlignCenter, label)

        font.setPointSize(10)
        font.setBold(True)
        qp.setFont(font)
        for xn, name, lv, hgt in self.peaks:
            x = self._x_of(xn)
            y = r.bottom() - hgt * ih
            col = readable(self.lut_q[int(lv * 255)], QtGui.QColor(th["bg"]).lightnessF() < 0.5)  # text on panel
            if self.mode == "curve":
                qp.setPen(QtGui.QPen(QtGui.QColor(th["panel"]), 1.5))
                qp.setBrush(col)
                qp.drawEllipse(QtCore.QPointF(x, y), 3.5, 3.5)
            qp.setPen(col)
            qp.drawText(QtCore.QRectF(x - 30, max(2.0, y - 22), 60, 18), QtCore.Qt.AlignCenter, name)
        qp.end()


class LevelMeter(QtWidgets.QWidget):
    """Level meter beside the spectrum, on the same dB scale as its axis.

    The bar shows the loudest bin between the walls (the value the threshold is
    compared against) with a slowly falling peak-hold tick. Two draggable lines set
    the colour range: the lower one is the threshold, the upper one is where the
    colours reach their brightest (at most 0 dBFS). Drag the nearer line, or scroll
    over it for 1 dB steps.
    """

    LABEL_H = GlowSpectrum.LABEL_H  # bottom strip for the caption; keeps the scale aligned with the spectrum
    RADIUS = 12
    HOLD_FALL = 0.4  # dB per frame
    WIDTH = 74
    MIN_SPAN = 6.0   # dB the two lines must stay apart

    thrMoved = QtCore.pyqtSignal(float, bool)  # threshold dBFS, drag finished
    topMoved = QtCore.pyqtSignal(float, bool)  # colour max dBFS, drag finished

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(self.WIDTH)
        self.setCursor(QtCore.Qt.SizeVerCursor)
        self.setToolTip("Loudest level between the walls (the bar), with a falling peak-hold mark.\n"
                        "Lower (pink) line: threshold – peaks below it are ignored.\n"
                        "Upper line: colour max – the level where the colours are brightest (at most 0 dB).\n"
                        "Drag the nearer line, or scroll over it for 1 dB steps.")
        self.floor, self.thr, self.top = -90.0, -60.0, 0.0
        self.level = self.hold = -120.0
        self._drag = None  # "thr" or "top"

    def set_theme(self, th, lut, lut_q, *_):
        self.th, self.lut_q = th, lut_q
        self.update()

    def set_levels(self, floor, thr, top):
        self.floor, self.thr, self.top = floor, thr, top
        self.update()

    def set_level(self, db):
        self.level = db
        self.hold = max(db, self.hold - self.HOLD_FALL)
        self.update()

    def _rect(self):
        return QtCore.QRectF(0, 0, self.width(), max(1, self.height() - self.LABEL_H))

    def _height(self, db):
        return float(np.clip((db - self.floor) / -self.floor, 0, 1))

    def _y(self, db):
        r = self._rect()
        return r.bottom() - self._height(db) * r.height()

    def _db_at(self, y):
        r = self._rect()
        return self.floor + (r.bottom() - y) / r.height() * -self.floor

    def _nearest(self, y):
        return "thr" if abs(y - self._y(self.thr)) <= abs(y - self._y(self.top)) else "top"

    def _move(self, which, db, finished):
        if which == "thr":
            self.thrMoved.emit(float(np.clip(round(db), self.floor, self.top - self.MIN_SPAN)), finished)
        else:
            self.topMoved.emit(float(np.clip(round(db), self.thr + self.MIN_SPAN, 0.0)), finished)

    def mousePressEvent(self, ev):
        if ev.button() == QtCore.Qt.LeftButton:
            self._drag = self._nearest(ev.pos().y())
            self._move(self._drag, self._db_at(ev.pos().y()), False)

    def mouseMoveEvent(self, ev):
        if self._drag:
            self._move(self._drag, self._db_at(ev.pos().y()), False)

    def mouseReleaseEvent(self, ev):
        if self._drag:
            self._move(self._drag, self.thr if self._drag == "thr" else self.top, True)
            self._drag = None

    def wheelEvent(self, ev):
        which = self._nearest(ev.pos().y())
        step = 1 if ev.angleDelta().y() > 0 else -1
        self._move(which, (self.thr if which == "thr" else self.top) + step, True)

    def _paint_line(self, qp, bar, w, r, db, color, label_above):
        y = self._y(db)
        qp.setPen(QtGui.QPen(color, 1))
        qp.drawLine(QtCore.QPointF(6, y), QtCore.QPointF(w - 6, y))
        handle = QtGui.QPolygonF([QtCore.QPointF(bar.right() + 2, y),
                                  QtCore.QPointF(bar.right() + 10, y - 5),
                                  QtCore.QPointF(bar.right() + 10, y + 5)])
        qp.setPen(QtCore.Qt.NoPen)
        qp.setBrush(color)
        qp.drawPolygon(handle)
        qp.setPen(color)
        ty = float(np.clip(y - 18 if label_above else y + 4, 2, r.bottom() - 16))
        qp.drawText(QtCore.QRectF(bar.right() + 6, ty, w - bar.right() - 8, 14),
                    QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter, f"{db:.0f}")

    def paintEvent(self, ev):
        qp = QtGui.QPainter(self)
        qp.setRenderHint(QtGui.QPainter.Antialiasing, True)
        w, h = self.width(), self.height()
        th = self.th
        clip = QtGui.QPainterPath()
        clip.addRoundedRect(QtCore.QRectF(0, 0, w, h), self.RADIUS, self.RADIUS)
        qp.setClipPath(clip)  # no fill of its own: the card behind is the background
        r = self._rect()

        # bar: track, level fill (same colours as the spectrum), 10 dB marks, peak hold
        bar = QtCore.QRectF(12, r.top() + 2, 14, r.height() - 4)
        track = QtGui.QPainterPath()
        track.addRoundedRect(bar, 4, 4)
        qp.fillPath(track, QtGui.QColor(th["button"]))
        y_lv = self._y(self.level)
        qp.save()
        qp.setClipPath(track, QtCore.Qt.IntersectClip)
        qp.fillRect(QtCore.QRectF(bar.left(), y_lv, bar.width(), bar.bottom() - y_lv),
                    level_gradient(self.lut_q, r, self._height(self.thr), self._height(self.top)))
        qp.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 110), 2, QtCore.Qt.SolidLine, QtCore.Qt.FlatCap))
        for db in np.arange(-10, self.floor + 1e-6, -10):
            y = round(self._y(db))
            qp.drawLine(QtCore.QPointF(bar.left(), y), QtCore.QPointF(bar.right(), y))
        hold = QtGui.QColor(th["fg"])
        hold.setAlpha(190)
        y_h = self._y(self.hold)
        qp.setPen(QtGui.QPen(hold, 2))
        qp.drawLine(QtCore.QPointF(bar.left(), y_h), QtCore.QPointF(bar.right(), y_h))
        qp.restore()

        font = qp.font()
        font.setPixelSize(12)  # fixed pixel size so the values fit the narrow meter at any display scale
        font.setBold(True)
        qp.setFont(font)
        # colour max above (label under its line, so it stays visible at 0 dB), threshold below
        self._paint_line(qp, bar, w, r, self.top, QtGui.QColor(th["accent"]), label_above=False)
        self._paint_line(qp, bar, w, r, self.thr, QtGui.QColor(th["thr"]), label_above=True)

        font.setPixelSize(11)
        font.setBold(False)
        qp.setFont(font)
        qp.setPen(QtGui.QColor(th["dim"]))
        qp.drawText(QtCore.QRectF(0, r.bottom(), w, self.LABEL_H), QtCore.Qt.AlignCenter, "LEVEL")
        qp.end()


class TiltGauge(QtWidgets.QWidget):
    """Dashboard-style dial for the spectrum tilt (dB per octave around 1 kHz).

    Drag to turn it (right/up = more), scroll for 0.5 dB steps, double-click to reset to 0.
    """

    RANGE = 9.0       # dB/oct either side of 0
    STEP = 0.5
    SWEEP = 220.0     # degrees of arc, symmetric around 12 o'clock
    PX_PER_STEP = 6

    valueChanged = QtCore.pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(84, 84)
        self.setCursor(QtCore.Qt.SizeAllCursor)
        self.setToolTip("Spectrum tilt, in dB per octave around 1 kHz.\n"
                        "+ raises the lows and lowers the highs; – does the opposite.\n"
                        "Music has less energy up high, so a – tilt helps high notes reach the threshold.\n"
                        "Applies to detection too, not just the picture.\n"
                        "Drag or scroll to adjust, double-click to reset to 0.")
        self.value = 0.0
        self._drag = None

    def set_theme(self, th, lut, lut_q, *_):
        self.th = th
        self.update()

    def set_value(self, v, emit=True):
        v = float(np.clip(round(v / self.STEP) * self.STEP, -self.RANGE, self.RANGE))
        if v != self.value:
            self.value = v
            self.update()
            if emit:
                self.valueChanged.emit(v)

    def mousePressEvent(self, ev):
        if ev.button() == QtCore.Qt.LeftButton:
            self._drag = (ev.globalPos(), self.value)

    def mouseMoveEvent(self, ev):
        if self._drag is not None:
            d = ev.globalPos() - self._drag[0]
            self.set_value(self._drag[1] + (d.x() - d.y()) / self.PX_PER_STEP * self.STEP)

    def mouseReleaseEvent(self, ev):
        self._drag = None

    def mouseDoubleClickEvent(self, ev):
        self.set_value(0.0)

    def wheelEvent(self, ev):
        self.set_value(self.value + (self.STEP if ev.angleDelta().y() > 0 else -self.STEP))

    def _angle(self, v):
        """Qt angle in degrees (0 = 3 o'clock, counter-clockwise) for a gauge value."""
        return 90.0 - v / self.RANGE * self.SWEEP / 2

    def paintEvent(self, ev):
        qp = QtGui.QPainter(self)
        qp.setRenderHint(QtGui.QPainter.Antialiasing, True)
        th = self.th
        w = self.width()
        rad = w / 2 - 9
        cx, cy = w / 2, rad + 8
        arc = QtCore.QRectF(cx - rad, cy - rad, 2 * rad, 2 * rad)
        start = self._angle(-self.RANGE)

        # track, then the lit part from 0 to the value
        qp.setPen(QtGui.QPen(QtGui.QColor(th["button"]), 4, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap))
        qp.drawArc(arc, int(start * 16), int(-self.SWEEP * 16))
        if self.value != 0:
            g = QtGui.QConicalGradient(cx, cy, 90)
            g.setColorAt(0.0, QtGui.QColor(th["accent"]))
            g.setColorAt(0.5, QtGui.QColor(th["accent2"]))
            g.setColorAt(1.0, QtGui.QColor(th["accent"]))
            qp.setPen(QtGui.QPen(QtGui.QBrush(g), 4, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap))
            qp.drawArc(arc, int(90 * 16), int((self._angle(self.value) - 90) * 16))

        # ticks every 3 dB/oct (longer at 0 and the ends)
        for v in np.arange(-self.RANGE, self.RANGE + 0.01, 3.0):
            a = np.radians(self._angle(v))
            major = v in (0, -self.RANGE, self.RANGE)
            r0, r1 = rad - 7, rad - (12 if major else 10)
            col = QtGui.QColor(th["fg"] if major else th["dim"])
            qp.setPen(QtGui.QPen(col, 1.5 if major else 1))
            qp.drawLine(QtCore.QPointF(cx + r0 * np.cos(a), cy - r0 * np.sin(a)),
                        QtCore.QPointF(cx + r1 * np.cos(a), cy - r1 * np.sin(a)))

        # needle + hub
        a = np.radians(self._angle(self.value))
        tip = QtCore.QPointF(cx + (rad - 4) * np.cos(a), cy - (rad - 4) * np.sin(a))
        qp.setPen(QtGui.QPen(QtGui.QColor(th["thr"]), 2, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap))
        qp.drawLine(QtCore.QPointF(cx, cy), tip)
        qp.setPen(QtCore.Qt.NoPen)
        qp.setBrush(QtGui.QColor(th["fg"]))
        qp.drawEllipse(QtCore.QPointF(cx, cy), 3.5, 3.5)

        # readout under the hub
        font = qp.font()
        font.setPixelSize(13)
        font.setBold(True)
        qp.setFont(font)
        qp.setPen(QtGui.QColor(th["fg"]))
        text_y = cy + rad * np.sin(np.radians(self.SWEEP / 2 - 90)) + 3  # just below the arc's ends
        qp.drawText(QtCore.QRectF(0, text_y, w, 16), QtCore.Qt.AlignCenter,
                    "0" if self.value == 0 else f"{self.value:+.1f}")
        font.setPixelSize(9)
        font.setBold(False)
        qp.setFont(font)
        qp.setPen(QtGui.QColor(th["dim"]))
        qp.drawText(QtCore.QRectF(0, text_y + 16, w, 12), QtCore.Qt.AlignCenter, "TILT")
        qp.end()


class Bubble(QtWidgets.QLabel):
    """A small pill-shaped label drawn over the window, placed exactly where it is told.

    Used for the Decay value under the slider and for short notices at the bottom of the window;
    unlike a tooltip it never moves itself, so it doesn't jump while it follows something.
    """

    def __init__(self, parent):
        super().__init__(parent)
        self.setObjectName("bubble")
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self.hide()
        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.hide)

    def show_at(self, text, center_x, top_y, hide_after_ms=0):
        """Show `text` centred on center_x, top edge at top_y (both in parent coordinates)."""
        self.setText(text)
        self.adjustSize()
        self.move(int(center_x - self.width() / 2), int(top_y))
        self.show()
        self.raise_()
        if hide_after_ms:
            self._timer.start(hide_after_ms)
        else:
            self._timer.stop()

    def hide_later(self, ms):
        self._timer.start(ms)


class WindowFrame(QtWidgets.QWidget):
    """Central widget of the frameless window: paints the (see-through) window background with rounded
    corners and lets the outer margin resize the window like a normal frame."""

    EDGE = 8  # px of margin that act as a resize border

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.color = QtGui.QColor("#1c1c1f")

    def _edges(self, pos):
        if self.window().isMaximized():
            return QtCore.Qt.Edges()
        e, r = QtCore.Qt.Edges(), self.rect()
        if pos.x() <= self.EDGE:
            e |= QtCore.Qt.LeftEdge
        if pos.x() >= r.width() - self.EDGE:
            e |= QtCore.Qt.RightEdge
        if pos.y() <= self.EDGE:
            e |= QtCore.Qt.TopEdge
        if pos.y() >= r.height() - self.EDGE:
            e |= QtCore.Qt.BottomEdge
        return e

    def mouseMoveEvent(self, ev):
        e = self._edges(ev.pos())
        shapes = {
            int(QtCore.Qt.LeftEdge | QtCore.Qt.TopEdge): QtCore.Qt.SizeFDiagCursor,
            int(QtCore.Qt.RightEdge | QtCore.Qt.BottomEdge): QtCore.Qt.SizeFDiagCursor,
            int(QtCore.Qt.RightEdge | QtCore.Qt.TopEdge): QtCore.Qt.SizeBDiagCursor,
            int(QtCore.Qt.LeftEdge | QtCore.Qt.BottomEdge): QtCore.Qt.SizeBDiagCursor,
            int(QtCore.Qt.LeftEdge): QtCore.Qt.SizeHorCursor, int(QtCore.Qt.RightEdge): QtCore.Qt.SizeHorCursor,
            int(QtCore.Qt.TopEdge): QtCore.Qt.SizeVerCursor, int(QtCore.Qt.BottomEdge): QtCore.Qt.SizeVerCursor,
        }
        if int(e) in shapes:
            self.setCursor(shapes[int(e)])
        else:
            self.unsetCursor()

    def mousePressEvent(self, ev):
        e = self._edges(ev.pos())
        if ev.button() == QtCore.Qt.LeftButton and int(e):
            self.window().windowHandle().startSystemResize(e)

    def paintEvent(self, ev):
        qp = QtGui.QPainter(self)
        qp.setRenderHint(QtGui.QPainter.Antialiasing, True)
        radius = 0 if self.window().isMaximized() else 14
        qp.setPen(QtCore.Qt.NoPen)
        qp.setBrush(self.color)
        qp.drawRoundedRect(QtCore.QRectF(self.rect()), radius, radius)
        qp.end()


class TitleBar(QtWidgets.QWidget):
    """Slim title bar for the frameless window: drag to move (with Windows snapping), double-click to
    maximise, and minimise / maximise / close buttons."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(26)
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(6, 0, 0, 0)
        lay.setSpacing(4)
        self.label = QtWidgets.QLabel(f"PitchView {VERSION}")
        self.label.setProperty("caption", True)
        lay.addWidget(self.label)
        lay.addStretch(1)
        for name, text, slot in (("winMin", "–", lambda: self.window().showMinimized()),
                                 ("winMax", "□", self._toggle_max),
                                 ("winClose", "✕", lambda: self.window().close())):
            b = QtWidgets.QToolButton()
            b.setObjectName(name)
            b.setText(text)
            b.setFixedSize(30, 22)
            b.clicked.connect(slot)
            lay.addWidget(b)

    def _toggle_max(self):
        w = self.window()
        w.showNormal() if w.isMaximized() else w.showMaximized()

    def mousePressEvent(self, ev):
        if ev.button() == QtCore.Qt.LeftButton:
            self.window().windowHandle().startSystemMove()

    def mouseDoubleClickEvent(self, ev):
        self._toggle_max()


class TitleMark(QtWidgets.QWidget):
    """'PITCHVIEW' wordmark filled with the theme's accent gradient."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(30)
        self.setMinimumWidth(170)

    def set_theme(self, th, *_):
        self.th = th
        self.update()

    def paintEvent(self, ev):
        qp = QtGui.QPainter(self)
        qp.setRenderHint(QtGui.QPainter.Antialiasing, True)
        font = QtGui.QFont(UI_FONT, 17)
        font.setWeight(QtGui.QFont.Medium)
        font.setLetterSpacing(QtGui.QFont.AbsoluteSpacing, 2)
        path = QtGui.QPainterPath()
        path.addText(0, 24, font, "PITCHVIEW")
        br = path.boundingRect()
        g = QtGui.QLinearGradient(br.left(), 0, br.right(), 0)
        g.setColorAt(0, QtGui.QColor(self.th["accent"]))
        g.setColorAt(1, QtGui.QColor(self.th["accent2"]))
        qp.fillPath(path, QtGui.QBrush(g))
        qp.end()


# hover explanations, keyed by the setting's label
TIPS = {
    "View": "How the spectrum is drawn.\n"
            "Prism – glow lines filling the spectrum's peaks\n"
            "Beam – glow lines only\n"
            "Line – plain lines; lightest to draw, and shows the threshold line",
    "Range": "Relative threshold: hide peaks more than this many dB below the loudest peak right now.\n"
             "The threshold on the level meter is a fixed line; Range follows the music instead.\n"
             "Loud passages: drops the small leftovers around big notes.\n"
             "Quiet passages: the cut-off moves down too, so soft notes still show.\n"
             "0 = off (threshold only).",
    "Tuning tolerance": "Only count peaks within this many cents of a piano key (100 cents = 1 semitone).\n"
                        "Noise, drums and out-of-tune overtones land between keys, so smaller = cleaner notes.\n"
                        "50 = off. If the music isn't tuned to A4 = 440 Hz, set Settings → Advanced → A4 reference.",
    "Min note length": "Notes shorter than this are not drawn on the piano.\n"
                       "Filters out flickers and short noise; once a note is confirmed its bar starts at the real onset.\n"
                       "0 = off.",
    "Decay": "How slowly the spectrum falls back after a sound stops.\nLeft = instant, right = slow and smooth.",
    "Harmonics": "A played note also makes peaks at 2×, 3×, 4×… its frequency (overtones).\n"
                 "Hide on: those weaker overtone peaks are hidden and only the note itself is kept.\n"
                 "An octave played together with the note below it may be merged into one.",
    "Display": "Freeze the spectrum and the piano roll (Space).",
    "Settings": "Theme, input device, FFT size, A4 reference and dB floor.",
    "Roll": "What the piano roll shows.\n"
            "Keys – detected notes, one lane per key\n"
            "Freq – the spectrum itself scrolling up, Wave Candy style",
    "Notes": "Maximum number of notes detected at the same time; the loudest ones are kept.",
    "Speed": "How fast the piano roll scrolls up. 1x = 240 pixels per second.",
    "Theme": "Dark or white window; the loudness colours are set by Gradient.",
    "Gradient": "Colours used for loudness (spectrum, meter, piano, note names).\n"
                "↻ makes a new random one (and switches to Random).",
    "Background": "How solid the window background is. Lower = see-through, so PitchView can sit over a video "
                  "or another window; the spectrum, notes and keys stay fully visible.",
    "dB floor": "Bottom of the dB axis for the spectrum and the level meter (the top is always 0 dB).",
    "Input": "Where the sound comes from.\n🔊 PC audio = whatever the PC is playing (speakers / headphones)\n"
             "🎤 Mic = a microphone, e.g. to watch a real piano",
    "FFT size": "Length of the analysis window.\n"
                "Bigger = finer resolution for low notes but slower response. 8192–16384 suits piano.",
    "A4 reference": "Tuning reference for note names.\n"
                    "Change it if the music isn't tuned to 440 Hz: when the ¢ readout keeps leaning one way, "
                    "move A4 that way.",
}

# spectrum view modes: (UI name, GlowSpectrum mode, tooltip)
SPEC_MODES = [
    ("Prism", "both", "Glow lines filling the spectrum's peaks"),
    ("Beam", "glow", "Glow lines only"),
    ("Line", "curve", "Lines only · lightest"),
]


class MainWindow(QtWidgets.QMainWindow):
    MAX_LABELS = 12
    SPEC_GAP = 6  # px between the spectrum and the level meter
    TITLE_MIN_W = 240  # px for the title column (wordmark + readout lines)
    NOTE_BOX = 92      # px: the big note readout is a square this size
    CTRL_H = 20        # px: height of every control in the header
    FRAME_MS = 16     # target frame time (~60 fps)
    DEFAULTS_REV = 12  # latest one-time defaults update in _load_settings
    ROLL_SHARPEN_RADIUS = 6  # px around each column searched for its peak's top in the freq roll
    ROLL_FLOOR = 0.18        # share of brightness kept away from peak tops (the faint continuous layer)
    ROLL_BASS_WIDTH = 8.0    # px (at half brightness) of freq-roll lines below C4: how wide C4-C5 notes look
    ROLL_BASS_BLEND = (60, 67)  # MIDI notes (C4, G4) between which bass lines blend into the normal look
    MIN_IDLE_MS = 6   # idle time left for input after every frame

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"PitchView {VERSION}")
        if os.path.exists(ICON_PATH):
            self.setWindowIcon(QtGui.QIcon(ICON_PATH))
        self.resize(1400, 900)
        load_fonts()
        app = QtWidgets.QApplication.instance()
        for effect in (QtCore.Qt.UI_AnimateCombo, QtCore.Qt.UI_AnimateMenu, QtCore.Qt.UI_FadeMenu,
                       QtCore.Qt.UI_AnimateTooltip, QtCore.Qt.UI_FadeTooltip):
            app.setEffectEnabled(effect, False)  # popups open at once instead of sliding in
        self.settings = QtCore.QSettings("PitchView", "PitchView")
        self.p = pa.PyAudio()
        self.capture = None
        self.analyzer = Analyzer()
        self.paused = False
        self.last_note = None
        self.last_note_time = 0.0
        self.floor_db = -90.0  # bottom of the dB axis; set in the settings dialog, not by the threshold
        n_keys = PIANO_HI - PIANO_LO + 1
        self.key_since = np.zeros(n_keys)          # onset time of the current note
        self.key_last = np.full(n_keys, -1e9)      # last time the key was detected
        self.key_frames = np.zeros(n_keys, int)    # frames since onset
        self.key_level = np.full(n_keys, -1, np.float32)
        self.key_shown = np.zeros(n_keys, bool)

        self._build_ui()
        self._populate_devices()

        # frames are chained, not fixed-rate: after each frame the event loop always gets some idle
        # time, so clicks and dropdowns stay responsive even when a heavy view can't make 60 fps
        self.timer = QtCore.QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self._frame)
        self._frame_start = None
        self._frame_delay = self.FRAME_MS
        self.timer.start(self.FRAME_MS)

        geom = self.settings.value("geometry")
        if geom is not None:
            self.restoreGeometry(geom)
        self._watch_source()
        QtWidgets.QApplication.instance().installEventFilter(self)  # click anywhere -> leave the text field

    def eventFilter(self, obj, ev):
        if ev.type() == QtCore.QEvent.MouseButtonPress:
            focused = QtWidgets.QApplication.focusWidget()
            # a spin box keeps the focus itself and forwards it to its inner line edit
            edit = focused.lineEdit() if isinstance(focused, QtWidgets.QAbstractSpinBox) else focused
            if isinstance(edit, QtWidgets.QLineEdit) and obj not in (focused, edit):
                edit.deselect()
                focused.clearFocus()
        return False

    # ---- UI -----------------------------------------------------------------
    def _build_ui(self):
        # frameless and see-through: the window draws its own background, title bar and resize border
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        central = self.frame = WindowFrame()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(12, 4, 12, 12)
        root.setSpacing(12)
        self.titlebar = TitleBar()
        root.addWidget(self.titlebar)
        root.setSpacing(6)

        def card(layout_cls=QtWidgets.QVBoxLayout, margins=10):
            f = QtWidgets.QFrame()
            f.setObjectName("card")
            lay = layout_cls(f)
            lay.setContentsMargins(margins, margins, margins, margins)
            return f, lay

        left = QtWidgets.QVBoxLayout()
        left.setSpacing(12)  # between the cards (the title bar sits 6 px above)
        root.addLayout(left, 1)

        # header card: wordmark + note readout + every everyday control + tilt gauge
        head_card, head = card(QtWidgets.QHBoxLayout, 10)
        head.setContentsMargins(12, 10, 12, 10)
        head.setSpacing(14)
        self.title = TitleMark()
        self.note_label = QtWidgets.QLabel("—")
        self.note_label.setAlignment(QtCore.Qt.AlignCenter)  # the note sits centred in its square box
        self.info_label = QtWidgets.QLabel("")
        self.chord_label = QtWidgets.QLabel("")
        for lab in (self.info_label, self.chord_label):
            # never ask the layout for more room: long text is cut to fit instead of widening the window
            lab.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
            lab.setMinimumWidth(0)
        title_box = QtWidgets.QWidget()
        title_box.setMinimumWidth(self.TITLE_MIN_W)  # room for the whole "Hz · ¢ · dBFS" readout
        title_col = QtWidgets.QVBoxLayout(title_box)
        title_col.setContentsMargins(0, 0, 0, 0)
        title_col.setSpacing(0)
        title_col.addWidget(self.title)
        title_col.addWidget(self.info_label)
        title_col.addWidget(self.chord_label)
        title_col.addStretch(1)
        head.addWidget(self.note_label)
        head.addWidget(title_box, 1)

        self.mode_combo = QtWidgets.QComboBox()
        for i, (name, _, tip) in enumerate(SPEC_MODES):
            self.mode_combo.addItem(name)
            self.mode_combo.setItemData(i, tip, QtCore.Qt.ToolTipRole)
        self.rel_spin = QtWidgets.QDoubleSpinBox()
        self.rel_spin.setDecimals(0)
        self.rel_spin.setRange(0, 80)
        self.rel_spin.setSuffix(" dB")
        self.npeaks_spin = QtWidgets.QSpinBox()
        self.npeaks_spin.setRange(1, self.MAX_LABELS)
        self.smooth_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.smooth_slider.setRange(0, 100)  # position; the decay follows a curve, see _decay()
        self.smooth_slider.sliderPressed.connect(self._show_decay_value)
        self.smooth_slider.valueChanged.connect(self._show_decay_value)
        self.smooth_slider.sliderReleased.connect(lambda: self.decay_bubble.hide_later(700))
        self.opacity_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.opacity_slider.setRange(30, 100)  # % background opacity
        self.opacity_slider.setPageStep(10)
        self.opacity_slider.valueChanged.connect(self._apply_theme)
        self.opacity_slider.sliderPressed.connect(self._show_opacity_value)
        self.opacity_slider.valueChanged.connect(self._show_opacity_value)
        self.opacity_slider.sliderReleased.connect(lambda: self.decay_bubble.hide_later(700))
        self.tune_spin = QtWidgets.QSpinBox()
        self.tune_spin.setRange(5, 50)
        self.tune_spin.setSuffix(" ¢")
        self.mindur_spin = QtWidgets.QSpinBox()
        self.mindur_spin.setRange(0, 1000)
        self.mindur_spin.setSingleStep(10)
        self.mindur_spin.setSuffix(" ms")
        self.drop_harm = QtWidgets.QCheckBox("Hide")
        self.pause_btn = QtWidgets.QPushButton("Pause")
        self.pause_btn.setCheckable(True)
        self.pause_btn.toggled.connect(self._set_paused)
        advanced_btn = self.advanced_btn = QtWidgets.QPushButton("Advanced")
        advanced_btn.clicked.connect(self._open_settings)

        def cell(caption, widget):
            box = QtWidgets.QWidget()
            lay = QtWidgets.QVBoxLayout(box)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setSpacing(4)  # caption -> control
            cap = QtWidgets.QLabel(caption)
            cap.setProperty("caption", True)
            lay.addWidget(cap)
            lay.addWidget(widget)
            return box

        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(6)
        cells = [
            ("View", self.mode_combo), ("Range", stepper(self.rel_spin)),
            ("Min note length", stepper(self.mindur_spin)), ("Display", self.pause_btn),
            ("Harmonics", self.drop_harm), ("Decay", self.smooth_slider),
            ("Background", self.opacity_slider), ("Settings", advanced_btn),
        ]
        for i, (caption, widget) in enumerate(cells):
            grid.addWidget(cell(caption, widget), i // 4, i % 4)
        for col in range(4):
            grid.setColumnStretch(col, 1)  # columns share the room; no fixed minimum, so the window can shrink
        for _, widget in cells:
            # number fields and dropdowns need room for the value plus their arrows
            widget.setMinimumWidth(86 if isinstance(widget, (QtWidgets.QAbstractSpinBox, QtWidgets.QComboBox)) else 70)
            widget.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        for _, widget in cells:
            widget.setProperty("hdr", True)  # header controls share one height (stylesheet rule "hdr")
        for widget in (self.smooth_slider, self.opacity_slider, self.drop_harm):  # no box of their own: centre them in the same row height
            widget.setFixedHeight(self.CTRL_H)
        head.addLayout(grid, 2)

        self.tilt_gauge = TiltGauge()
        self.tilt_gauge.valueChanged.connect(self._tilt_changed)
        head.addWidget(self.tilt_gauge, 0, QtCore.Qt.AlignVCenter)
        left.addWidget(head_card)

        # spectrum card: spectrum (any mode) + level meter (threshold and colour range)
        self.spec_card, spec_lay = card(QtWidgets.QHBoxLayout, 8)
        spec_lay.setSpacing(self.SPEC_GAP)
        self.glow = GlowSpectrum()
        self.glow.wallMoved.connect(self._glow_wall_moved)
        spec_lay.addWidget(self.glow, 1)
        self.meter = LevelMeter()
        self.meter.thrMoved.connect(self._thr_moved)
        self.meter.topMoved.connect(self._top_moved)
        spec_lay.addWidget(self.meter)
        # spectrum and piano share a splitter: drag the gap between them to trade height
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setHandleWidth(12)
        self.splitter.addWidget(self.spec_card)
        left.addWidget(self.splitter, 1)

        self.roll_card, lay = card(QtWidgets.QHBoxLayout, 8)
        # same horizontal insets as the spectrum's plot area, so C1, C2, ... line up with the keys;
        # the column under the level meter holds the piano's own settings
        lay.setContentsMargins(8 + GlowSpectrum.AXIS_W, 8, 8, 8)
        lay.setSpacing(self.SPEC_GAP)
        self.roll = PianoRoll()
        lay.addWidget(self.roll, 1)
        self.roll_speed = QtWidgets.QDoubleSpinBox()
        self.roll_speed.setRange(0.25, 8)
        self.roll_speed.setSingleStep(0.25)
        self.roll_speed.setDecimals(2)
        self.roll_speed.setSuffix("x")
        self.roll_speed.valueChanged.connect(lambda v: setattr(self.roll, "speed", v))
        side = QtWidgets.QWidget()
        side.setFixedWidth(LevelMeter.WIDTH)
        side_lay = QtWidgets.QVBoxLayout(side)
        side_lay.setContentsMargins(0, 4, 0, 0)
        side_lay.setSpacing(3)
        self.roll_mode_btn = QtWidgets.QPushButton("Keys")
        self.roll_mode_btn.setObjectName("pill")
        self.roll_mode_btn.setCheckable(True)
        self.roll_mode_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.roll_mode_btn.toggled.connect(self._roll_mode_changed)
        cap = QtWidgets.QLabel("Roll")
        cap.setProperty("caption", True)
        cap.setAlignment(QtCore.Qt.AlignCenter)
        side_lay.addWidget(cap)
        side_lay.addWidget(self.roll_mode_btn)
        side_lay.addSpacing(8)
        for caption, spin in (("Notes", self.npeaks_spin), ("Speed", self.roll_speed)):
            cap = QtWidgets.QLabel(caption)
            cap.setProperty("caption", True)
            cap.setAlignment(QtCore.Qt.AlignCenter)
            side_lay.addWidget(cap)
            side_lay.addWidget(stepper(spin, compact=True))
            side_lay.addSpacing(8)
        side_lay.addStretch(1)
        lay.addWidget(side)
        self.splitter.addWidget(self.roll_card)
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 2)

        # expand buttons at the top-left of each plot: one view takes the whole area, press again to restore
        self.solo = None
        self.expand_btns = {}
        for key, card_w, tip in (("spec", self.spec_card, "Spectrum only"), ("roll", self.roll_card, "Piano only")):
            b = QtWidgets.QToolButton(card_w)
            b.setObjectName("expand")
            b.setToolTip(tip)
            b.setCursor(QtCore.Qt.PointingHandCursor)
            b.setFixedSize(18, 18)
            b.move(8, 8)  # top-left corner of each card (same gap from the top and the left edge)
            b.clicked.connect(lambda _=False, k=key: self._toggle_solo(k))
            b.raise_()
            self.expand_btns[key] = b

        # detailed settings live in a separate dialog opened from "Advanced settings"
        self.settings_dlg = QtWidgets.QDialog(self)
        self.settings_dlg.setObjectName("settingsDialog")
        self.settings_dlg.setWindowTitle("Settings")
        self.settings_dlg.setMinimumWidth(380)
        sform = QtWidgets.QFormLayout(self.settings_dlg)
        sform.setContentsMargins(20, 12, 20, 20)
        sform.setVerticalSpacing(8)

        def section(text):
            lab = QtWidgets.QLabel(text)
            lab.setProperty("section", True)
            sform.addRow(lab)

        section("Display")
        self.theme_combo = QtWidgets.QComboBox()
        self.theme_combo.addItems(list(THEMES))
        sform.addRow("Theme", self.theme_combo)
        grad_row = QtWidgets.QHBoxLayout()
        self.grad_combo = QtWidgets.QComboBox()
        self.grad_combo.addItems(GRADIENT_CHOICES)
        self.grad_combo.currentIndexChanged.connect(self._gradient_changed)
        self.reroll_btn = QtWidgets.QPushButton()
        self.reroll_btn.setObjectName("roundIcon")
        self.reroll_btn.setToolTip("New random gradient")
        self.reroll_btn.clicked.connect(self._reroll_gradient)
        grad_row.setSpacing(8)
        grad_row.addWidget(self.grad_combo, 1)
        grad_row.addWidget(self.reroll_btn)
        sform.addRow("Gradient", grad_row)
        self.floor_spin = QtWidgets.QSpinBox()
        self.floor_spin.setRange(-150, -30)
        self.floor_spin.setSingleStep(5)
        self.floor_spin.setSuffix(" dB")
        sform.addRow("dB floor", stepper(self.floor_spin))

        section("Audio")
        dev_row = QtWidgets.QHBoxLayout()
        self.dev_combo = QtWidgets.QComboBox()
        self.dev_combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.dev_combo.setMinimumContentsLength(12)
        self.dev_combo.currentIndexChanged.connect(self._device_changed)
        self.refresh_btn = QtWidgets.QPushButton()
        self.refresh_btn.setObjectName("roundIcon")
        self.refresh_btn.setToolTip("Rescan audio devices (after plugging in or removing one)")
        self.refresh_btn.clicked.connect(self._rescan_devices)
        dev_row.setSpacing(8)
        dev_row.addWidget(self.dev_combo, 1)
        dev_row.addWidget(self.refresh_btn)
        sform.addRow("Input", dev_row)
        self.fft_combo = QtWidgets.QComboBox()
        for n in (2048, 4096, 8192, 16384, 32768):
            self.fft_combo.addItem(str(n), n)
        sform.addRow("FFT size", self.fft_combo)

        section("Pitch")
        self.a4_spin = QtWidgets.QDoubleSpinBox()
        self.a4_spin.setRange(400, 480)
        self.a4_spin.setDecimals(1)
        self.a4_spin.setSingleStep(0.5)
        self.a4_spin.setSuffix(" Hz")
        sform.addRow("A4 reference", stepper(self.a4_spin))
        sform.addRow("Tuning tolerance", stepper(self.tune_spin))

        reset_btn = QtWidgets.QPushButton("Reset all settings")
        reset_btn.clicked.connect(self._reset_settings)
        sform.addRow(QtWidgets.QLabel(""))
        sform.addRow(reset_btn)

        QtWidgets.QShortcut(QtGui.QKeySequence("Space"), self, activated=self.pause_btn.toggle)

        for combo in self.findChildren(QtWidgets.QComboBox):
            self._polish_combo(combo)
        self._show_idle()
        self.decay_bubble = Bubble(self)
        self.toast = Bubble(self)
        self._apply_tips()
        self._load_settings()
        self.roll.speed = self.roll_speed.value()
        self._roll_mode_changed(self.roll_mode_btn.isChecked())
        self._restore_split()

        for w in (self.a4_spin, self.floor_spin):
            sig = w.currentIndexChanged if isinstance(w, QtWidgets.QComboBox) else w.valueChanged
            sig.connect(self._update_axes)
        self.theme_combo.currentIndexChanged.connect(self._apply_theme)
        self.mode_combo.currentIndexChanged.connect(self._apply_mode)
        self._apply_theme()
        self._apply_mode()
        self._update_axes()
        self._sync_walls()

    # ---- settings -----------------------------------------------------------
    SETTINGS = [  # (attr, key, default)
        ("theme_combo", "theme", 0), ("mode_combo", "view_mode", 0),
        ("fft_combo", "fft", 2),
        ("rel_spin", "rel", 18.0),
        ("npeaks_spin", "npeaks", 6), ("roll_mode_btn", "roll_freq", False), ("roll_speed", "roll_speed", 1.0), ("mindur_spin", "mindur", 0), ("tune_spin", "tune", 30), ("smooth_slider", "smooth", 75), ("drop_harm", "harm", True),
        ("a4_spin", "a4", 440.0), ("grad_combo", "gradient", GRADIENT_CHOICES.index("Glacier")), ("floor_spin", "floor", -90), ("opacity_slider", "opacity", 100),
    ]

    def _load_settings(self):
        s = self.settings
        if s.value("defaults_rev", 0, type=int) < 1:
            # one-time switch to the new defaults, even over a previously saved choice
            s.setValue("view_mode", 2)  # Line
            s.setValue("thr", -60.0)
            s.setValue("defaults_rev", 1)
        if s.value("defaults_rev", 0, type=int) < 2:
            s.setValue("top", 0.0)  # colour max moved onto the meter, starting at 0 dBFS
            s.setValue("defaults_rev", 2)
        if s.value("defaults_rev", 0, type=int) < 3:
            s.setValue("roll_speed", 4.0)  # faster default piano scroll
            s.setValue("defaults_rev", 3)
        if s.value("defaults_rev", 0, type=int) < 4:
            s.setValue("top", -10.0)  # colour max line
            s.remove("fmin")  # walls back to their default spot near the edges
            s.remove("fmax")
            s.setValue("defaults_rev", 4)
        if s.value("defaults_rev", 0, type=int) < 5:
            # 1x now means what 4x used to; keep the same on-screen speed
            s.setValue("roll_speed", s.value("roll_speed", 4.0, type=float) / 4.0)
            s.setValue("defaults_rev", 5)
        if s.value("defaults_rev", 0, type=int) < 6:
            s.setValue("tilt", 1.5)
            s.setValue("defaults_rev", 6)
        if s.value("defaults_rev", 0, type=int) < 7 and s.contains("theme"):
            # themes became Dark / White; their colours moved into the gradient list
            old_theme = ['Chroma', 'Aurora', 'Ember', 'Paper'][min(s.value("theme", 0, type=int), 3)]
            old_grad = ['Theme', 'Magma', 'Viridis', 'Ocean', 'Sunset', 'Forest', 'Ice', 'Fire', 'Neon', 'Rose', 'Mono', 'Random'][min(s.value("gradient", 0, type=int), 11)]
            grad = old_theme if old_grad == "Theme" else old_grad
            grad = {"Chroma": "Twilight", "Ice": "Glacier", "Rose": "Blossom", "Mono": "Pastel"}.get(grad, grad)
            s.setValue("theme", 1 if old_theme == "Paper" else 0)
            s.setValue("gradient", GRADIENT_CHOICES.index(grad))
        if s.value("defaults_rev", 0, type=int) < 7:
            s.setValue("defaults_rev", 7)
        if s.value("defaults_rev", 0, type=int) < 8:
            s.setValue("gradient", GRADIENT_CHOICES.index("Aurora"))
            s.setValue("defaults_rev", 8)
        if s.value("defaults_rev", 0, type=int) < 9:
            # Decay slider became curved: move it to the position that gives the same decay as before
            if s.contains("smooth"):
                old = min(s.value("smooth", 60, type=int), 95) / 100.0
                pos = 1.0 - (1.0 - min(old / self.DECAY_MAX, 1.0)) ** (1 / self.DECAY_CURVE)
                s.setValue("smooth", int(round(pos * 100)))
            s.setValue("defaults_rev", 9)
        if s.value("defaults_rev", 0, type=int) < 10:
            s.setValue("smooth", 65)
            s.setValue("gradient", GRADIENT_CHOICES.index("Glacier"))  # Glacier also moved to the top of the list
            s.setValue("defaults_rev", 10)
        if s.value("defaults_rev", 0, type=int) < 11:
            s.setValue("smooth", 75)
            s.setValue("defaults_rev", 11)
        if s.value("defaults_rev", 0, type=int) < 12:
            s.setValue("tilt", 0.0)
            s.setValue("view_mode", 0)  # Prism
            s.setValue("defaults_rev", 12)
        if not s.contains("view_mode") and s.contains("spec_mode"):
            # old 4-mode list: Prism+glow, peaks only (removed), glow, curve
            s.setValue("view_mode", {0: 0, 1: 0, 2: 1, 3: 2}.get(s.value("spec_mode", 0, type=int), 0))
        self.view_lo, self.view_hi = piano_view_range(440.0)
        lo0, hi0 = default_walls(self.view_lo, self.view_hi)
        lo = float(np.clip(s.value("fmin", lo0, type=float), self.view_lo, self.view_hi / MIN_RATIO))
        hi = float(np.clip(s.value("fmax", hi0, type=float), lo * MIN_RATIO, self.view_hi))
        self.walls_hz = [lo, hi]
        self.thr_db = float(np.clip(s.value("thr", -60.0, type=float), -150, -6))
        self.tilt_db = float(np.clip(s.value("tilt", 0.0, type=float), -TiltGauge.RANGE, TiltGauge.RANGE))
        self.top_db = float(np.clip(s.value("top", -10.0, type=float), self.thr_db + LevelMeter.MIN_SPAN, 0.0))
        self.tilt_gauge.set_value(self.tilt_db, emit=False)
        for attr, key, default in self.SETTINGS:
            w = getattr(self, attr)
            v = s.value(key, default, type=type(default))
            if isinstance(w, QtWidgets.QComboBox):
                w.setCurrentIndex(v if 0 <= v < w.count() else default)
            elif isinstance(w, QtWidgets.QAbstractButton):
                w.setChecked(v)
            else:
                w.setValue(v)

    def _reset_settings(self):
        """Back to the defaults for everything (window size and position are kept)."""
        answer = QtWidgets.QMessageBox.question(
            self.settings_dlg, "Reset all settings",
            "Reset every setting to its default?\nThis can't be undone.",
            QtWidgets.QMessageBox.Reset | QtWidgets.QMessageBox.Cancel, QtWidgets.QMessageBox.Cancel)
        if answer != QtWidgets.QMessageBox.Reset:
            return
        s = self.settings
        geom = s.value("geometry")
        s.clear()
        s.setValue("defaults_rev", self.DEFAULTS_REV)  # defaults already current: skip the one-time updates
        if geom is not None:
            s.setValue("geometry", geom)
        self._load_settings()  # widgets fire their change signals, so most views follow on their own
        self.roll.speed = self.roll_speed.value()
        self._roll_mode_changed(self.roll_mode_btn.isChecked())
        self._restore_split()
        self._apply_theme()
        self._apply_mode()
        self._update_axes()
        self._notify("All settings reset to defaults")

    def _save_settings(self):
        s = self.settings
        s.setValue("fmin", self.walls_hz[0])
        s.setValue("fmax", self.walls_hz[1])
        s.setValue("thr", self.thr_db)
        s.setValue("split", self.splitter.saveState())
        s.setValue("input_name", self.dev_combo.currentText())
        s.setValue("tilt", self.tilt_db)
        s.setValue("top", self.top_db)
        s.setValue("geometry", self.saveGeometry())
        for attr, key, _ in self.SETTINGS:
            w = getattr(self, attr)
            if isinstance(w, QtWidgets.QComboBox):
                s.setValue(key, w.currentIndex())
            elif isinstance(w, QtWidgets.QAbstractButton):
                s.setValue(key, w.isChecked())
            else:
                s.setValue(key, w.value())

    # ---- theme / layout -----------------------------------------------------
    # where on the gradient each accent colour is taken from
    ACCENT_STOPS = {"accent": 0.85, "accent2": 0.55, "curve": 0.70, "thr": 0.35}

    def _apply_theme(self, *_):
        base = THEMES[self.theme_combo.currentText()]
        self.lut, self.lut_q, self.lut_rgb = build_lut(self._current_stops(base))
        # every accent (buttons, toggles, sliders, title, line view, gauge, meter lines) comes from the
        # chosen gradient; only the neutral window colours stay with the theme
        th = dict(base)
        dark = QtGui.QColor(base["bg"]).lightnessF() < 0.5
        for key, t in self.ACCENT_STOPS.items():
            th[key] = readable(self.lut_q[int(t * 255)], dark).name()
        c = QtGui.QColor(th["curve"])
        th["ghost"] = (c.red(), c.green(), c.blue(), 55)
        th["alpha"] = self.opacity_slider.value() / 100.0  # background opacity; content stays solid
        self.th = th
        self._idle_pix = None  # the idle glyph follows the gradient: redraw it
        if getattr(self, "_idle", False):
            self._show_idle()

        app = QtWidgets.QApplication.instance()
        pal = QtGui.QPalette()
        for role, key in [(QtGui.QPalette.Window, "panel"), (QtGui.QPalette.WindowText, "fg"),
                          (QtGui.QPalette.Base, "button"), (QtGui.QPalette.Text, "fg"),
                          (QtGui.QPalette.Button, "button"), (QtGui.QPalette.ButtonText, "fg"),
                          (QtGui.QPalette.ToolTipBase, "button"), (QtGui.QPalette.ToolTipText, "fg")]:
            pal.setColor(role, QtGui.QColor(th[key]))
        pal.setColor(QtGui.QPalette.Window, QtGui.QColor(th["bg"]))
        pal.setColor(QtGui.QPalette.Highlight, QtGui.QColor(th["accent"]))
        pal.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor(th["bg"]))
        app.setPalette(pal)
        self._update_expand_icons()
        refresh_icon = QtGui.QIcon(self._refresh_icon(th["fg"]))
        for btn in (self.refresh_btn, self.reroll_btn):
            btn.setIcon(refresh_icon)
            btn.setIconSize(QtCore.QSize(14, 14))
        chevron = self._chevron_icon(th["dim"])
        chevron_up = self._chevron_icon(th["dim"], up=True)
        a = th["alpha"]

        def see_through(key):
            q = QtGui.QColor(th[key])
            return f"rgba({q.red()}, {q.green()}, {q.blue()}, {a:.3f})"

        self.frame.color = QtGui.QColor(th["bg"])
        self.frame.color.setAlphaF(a)
        self.frame.update()
        self.setStyleSheet(f"""
            QMainWindow {{ background: transparent; color: {th['dim']}; }}
            QFrame#card {{ background: {see_through('panel')}; border-radius: 16px; }}
            QToolButton#winMin, QToolButton#winMax, QToolButton#winClose {{
                background: transparent; color: {th['dim']}; border: none; border-radius: 6px; font-size: 13px; }}
            QToolButton#winMin:hover, QToolButton#winMax:hover {{ background: {th['button']}; color: {th['fg']}; }}
            QToolButton#winClose:hover {{ background: #e5484d; color: #ffffff; }}
            QDialog#settingsDialog {{ background: {th['panel']}; }}
            QAbstractSpinBox {{ padding-right: 20px; }}
            QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {{
                subcontrol-origin: border; width: 18px; border: none; background: transparent; }}
            QAbstractSpinBox::up-button {{ subcontrol-position: top right; margin: 3px 4px 0 0; }}
            QAbstractSpinBox::down-button {{ subcontrol-position: bottom right; margin: 0 4px 3px 0; }}
            QAbstractSpinBox::up-button:hover, QAbstractSpinBox::down-button:hover {{
                background: {th['guide']}; border-radius: 5px; }}
            QAbstractSpinBox::up-arrow {{ image: url({chevron_up}); width: 8px; height: 8px; }}
            QAbstractSpinBox::down-arrow {{ image: url({chevron}); width: 8px; height: 8px; }}
            QAbstractSpinBox::up-arrow:pressed, QAbstractSpinBox::down-arrow:pressed {{ top: 1px; }}
            QAbstractSpinBox#compact {{ padding: 0 16px 0 0; font-size: 11px; border-radius: 10px;
                                        min-height: 24px; max-height: 24px; }}
            QAbstractSpinBox#compact::up-button {{ width: 14px; margin: 2px 3px 0 0; }}
            QAbstractSpinBox#compact::down-button {{ width: 14px; margin: 0 3px 2px 0; }}
            QComboBox[hdr="true"], QAbstractSpinBox[hdr="true"], QPushButton[hdr="true"] {{
                padding-top: 0; padding-bottom: 0; min-height: {self.CTRL_H}px; max-height: {self.CTRL_H}px;
                border-radius: {self.CTRL_H // 2}px; font-size: 11px; }}
            QAbstractSpinBox[hdr="true"]::up-button {{ width: 14px; margin: 1px 4px 0 0; }}
            QAbstractSpinBox[hdr="true"]::down-button {{ width: 14px; margin: 0 4px 1px 0; }}
            QAbstractSpinBox[hdr="true"]::up-arrow, QAbstractSpinBox[hdr="true"]::down-arrow {{ width: 7px; height: 7px; }}
            QComboBox[hdr="true"]::down-arrow {{ width: 8px; height: 8px; }}
            QCheckBox[hdr="true"] {{ font-size: 11px; }}
            QSplitter::handle {{ background: transparent; }}
            QSplitter::handle:hover {{ background: {th['guide']}; border-radius: 3px; margin: 4px 40%; }}
            QToolButton#expand {{ background: transparent; border: none; border-radius: 6px; }}
            QToolButton#expand:hover {{ background: {th['button']}; }}
            QPushButton#roundIcon {{ min-width: 30px; max-width: 30px; min-height: 30px; max-height: 30px;
                                     padding: 0; border-radius: 15px; }}
            QPushButton#pill {{ border-radius: 11px; min-height: 22px; max-height: 22px; padding: 0 8px;
                                font-size: 11px; }}
            QWidget#cardbody, QScrollArea, QScrollArea > QWidget > QWidget {{ background: {th['panel']}; }}
            QLabel {{ background: transparent; }}
            QLabel#bubble {{ background: {th['button']}; color: {th['fg']}; border: 1px solid {th['guide']};
                             border-radius: 9px; padding: 2px 9px; font-size: 11px; }}
            QLabel[section='true'] {{ color: {th['accent']}; font-weight: 700; letter-spacing: 1px;
                                      padding-top: 10px; }}
            QLabel[hint='true'] {{ color: {th['dim']}; }}
            QLabel[caption='true'] {{ color: {th['dim']}; font-size: 11px; }}
            QComboBox, QAbstractSpinBox, QPushButton {{
                background: {th['button']}; color: {th['fg']}; border: none;
                border-radius: 12px; padding: 2px 12px; min-height: 18px; font-size: 13px; }}
            QComboBox {{ combobox-popup: 0; }}  /* list opens below the box instead of over it */
            QComboBox::drop-down {{ border: none; width: 26px; }}
            QComboBox::down-arrow {{ image: url({chevron}); width: 10px; height: 10px; }}
            QComboBox::down-arrow:on {{ top: 1px; }}
            QComboBox QAbstractItemView {{ background: {th['button']}; color: {th['fg']};
                border: 1px solid {th['guide']}; border-radius: 10px; padding: 4px; outline: none; }}
            QComboBox QAbstractItemView::item {{ min-height: 26px; padding: 2px 10px; border-radius: 6px; }}
            QComboBox QAbstractItemView::item:hover {{ background: {th['guide']}; }}
            QComboBox QAbstractItemView::item:selected {{ background: {th['accent']}; color: {th['panel']}; }}
            QPushButton:hover, QComboBox:hover, QAbstractSpinBox:hover {{ background: {th['guide']}; }}
            QPushButton:checked {{ background: {th['accent']}; color: {th['panel']}; }}
            QSlider::groove:horizontal {{ height: 4px; background: {th['button']}; border-radius: 2px; }}
            QSlider::sub-page:horizontal {{ background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 {th['accent']}, stop:1 {th['accent2']}); border-radius: 2px; }}
            QSlider::handle:horizontal {{ background: {th['fg']}; width: 14px; margin: -6px 0;
                border-radius: 7px; }}
            QCheckBox {{ color: {th['fg']}; spacing: 8px; }}
            QCheckBox::indicator {{ width: 30px; height: 16px; border-radius: 8px;
                background: {th['button']}; }}
            QCheckBox::indicator:checked {{ background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 {th['accent']}, stop:1 {th['accent2']}); }}
            QScrollBar:vertical {{ background: transparent; width: 8px; }}
            QScrollBar::handle:vertical {{ background: {th['button']}; border-radius: 4px; }}
            QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
            QToolTip {{ background: {th['button']}; color: {th['fg']}; border: none; padding: 4px; }}
        """)
        self.info_label.setStyleSheet(f"font-size: 16px; color: {th['fg']};")
        self.chord_label.setStyleSheet(f"font-size: 20px; font-weight: 600; color: {th['dim']};")
        self._note_style(None)

        for w in (self.roll, self.glow, self.meter, self.title, self.tilt_gauge):
            w.set_theme(th, self.lut, self.lut_q, self.lut_rgb)

    @staticmethod
    def _expand_icon(color, collapse=False):
        """Two diagonal arrows: pointing out (expand) or in (collapse back)."""
        name = f"pitchview_{'collapse' if collapse else 'expand'}_{color.lstrip('#')}.svg"
        path = os.path.join(tempfile.gettempdir(), name)
        if not os.path.exists(path):
            if collapse:  # arrow heads at the inner ends
                d = "M2 14 L6.5 9.5 M3.5 9.5 L6.5 9.5 L6.5 12.5 M14 2 L9.5 6.5 M9.5 3.5 L9.5 6.5 L12.5 6.5"
            else:         # arrow heads at the outer corners
                d = "M9.5 6.5 L14 2 M10.5 2 L14 2 L14 5.5 M6.5 9.5 L2 14 M2 10.5 L2 14 L5.5 14"
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 16 16">'
                         f'<path d="{d}" fill="none" stroke="{color}" stroke-width="1.6" '
                         f'stroke-linecap="round" stroke-linejoin="round"/></svg>')
        return path

    @staticmethod
    def _refresh_icon(color):
        """Circular-arrow SVG in the theme colour, shared by every refresh / re-roll button."""
        path = os.path.join(tempfile.gettempdir(), f"pitchview_refresh_{color.lstrip('#')}.svg")
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 16 16">'
                         f'<g fill="none" stroke="{color}" stroke-width="1.7" stroke-linecap="round" '
                         f'stroke-linejoin="round"><path d="M13.2 8.6 A5.3 5.3 0 1 1 11.6 4.1"/>'
                         f'<path d="M12.2 1.6 L12.2 4.6 L9.2 4.6"/></g></svg>')
        return path

    @staticmethod
    def _chevron_icon(color, up=False):
        """Small chevron SVG in the theme colour for dropdown / number-field arrows (stylesheets need a file)."""
        name = f"pitchview_chevron_{'up' if up else 'down'}_{color.lstrip('#')}.svg"
        path = os.path.join(tempfile.gettempdir(), name)
        if not os.path.exists(path):
            d = "M1.5 6.5 L5 3 L8.5 6.5" if up else "M1.5 3.5 L5 7 L8.5 3.5"
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10" viewBox="0 0 10 10">'
                         f'<path d="{d}" fill="none" stroke="{color}" stroke-width="1.6" '
                         f'stroke-linecap="round" stroke-linejoin="round"/></svg>')
        return path.replace("\\", "/")

    @staticmethod
    def _polish_combo(combo):
        """Rounded, frameless popup list whose items follow the stylesheet."""
        combo.setItemDelegate(QtWidgets.QStyledItemDelegate(combo))  # lets ::item rules apply
        popup = combo.view().window()
        popup.setWindowFlags(popup.windowFlags() | QtCore.Qt.FramelessWindowHint | QtCore.Qt.NoDropShadowWindowHint)
        popup.setAttribute(QtCore.Qt.WA_TranslucentBackground)  # so the rounded corners show

    def _current_stops(self, th):
        """Colour-map stops: a preset, or the saved random one."""
        choice = self.grad_combo.currentText()
        if choice == "Random":
            colors = self.settings.value("random_gradient", "", type=str).split(",")
            if len(colors) != 5:
                colors = random_gradient()
                self.settings.setValue("random_gradient", ",".join(colors))
        else:
            colors = GRADIENTS.get(choice, GRADIENTS["Aurora"])
        if isinstance(colors[0], tuple):
            return colors
        return [(i / (len(colors) - 1), c) for i, c in enumerate(colors)]

    def _gradient_changed(self, *_):
        if hasattr(self, "th"):
            self._apply_theme()

    def _reroll_gradient(self):
        self.settings.setValue("random_gradient", ",".join(random_gradient()))
        if self.grad_combo.currentText() != "Random":
            self.grad_combo.setCurrentText("Random")  # applies the new colours through _gradient_changed
        else:
            self._apply_theme()

    def _restore_split(self):
        state = self.settings.value("split")
        if state is None or not self.splitter.restoreState(state):
            self.splitter.setSizes([600, 400])  # default 3 : 2

    def _check_mic(self, cap):
        """A microphone that delivers only exact zeros is muted, disconnected or blocked by Windows."""
        if cap is self.capture and not cap.heard:
            self._notify("No sound from this microphone – check it's connected and not muted "
                         "(Windows Settings → Sound / Privacy → Microphone)", 9000)

    def _toggle_solo(self, key):
        """Show one view alone (the other is hidden), or both again if it was already alone."""
        self.solo = None if self.solo == key else key
        self.spec_card.setVisible(self.solo != "roll")
        self.roll_card.setVisible(self.solo != "spec")
        self._update_expand_icons()

    def _update_expand_icons(self):
        if not hasattr(self, "th"):
            return
        for key, b in self.expand_btns.items():
            alone = self.solo == key
            b.setIcon(QtGui.QIcon(self._expand_icon(self.th["dim"], collapse=alone)))
            b.setIconSize(QtCore.QSize(10, 10))
            b.setToolTip("Back to both views" if alone else ("Spectrum only" if key == "spec" else "Piano only"))
            b.raise_()

    def _rescan_devices(self):
        self._populate_devices(rescan=True)
        if self.capture:
            self._notify(f"Rescanned: {self.dev_combo.count()} inputs found · listening to "
                         f"{self.dev_combo.currentText()}")

    def _apply_mode(self, *_):
        self.glow.set_mode(SPEC_MODES[self.mode_combo.currentIndex()][1])

    def _note_style(self, color):
        c = color.name() if color is not None else self.th["dim"]
        self.note_label.setStyleSheet(f"font-size: {getattr(self, 'note_px', 48)}px; font-weight: 300; color: {c};")
        self._fix_header_sizes()

    def _fix_header_sizes(self):
        """Fixed sizes for the readouts, so changing notes never nudge the header layout."""
        if getattr(self, "_header_fixed", False):
            return
        self._header_fixed = True
        side = self.NOTE_BOX
        self.note_label.setFixedSize(side, side)  # square: as tall as the header's content
        self.note_px = 60
        while self.note_px > 20:
            f = QtGui.QFont(self.note_label.font())
            f.setPixelSize(self.note_px)
            if max(QtGui.QFontMetrics(f).horizontalAdvance(t) for t in ("C#8", "G#8", "A#0")) <= side - 6:
                break
            self.note_px -= 2
        self._note_style(None)
        for lab in (self.info_label, self.chord_label):
            lab.ensurePolished()
            lab.setFixedHeight(lab.fontMetrics().height() + 4)
        self._header_fixed = True

    def _set_paused(self, on):
        self.paused = on
        self.roll.paused = on

    # ---- devices ------------------------------------------------------------
    def _populate_devices(self, rescan=False):
        """Fill the input list, keeping the chosen device when it is still there.

        PortAudio only enumerates devices when it starts, so a rescan restarts it; otherwise
        plugged-in or removed devices would never show up.
        """
        keep = self.dev_combo.currentText() if self.dev_combo.count() else \
            self.settings.value("input_name", "", type=str)
        if rescan:
            if self.capture:
                self.capture.stop()
                self.capture = None
            self.p.terminate()
            self.p = pa.PyAudio()
        self.dev_combo.blockSignals(True)
        self.dev_combo.clear()
        try:
            default_lb = self.p.get_default_wasapi_loopback()["index"]
        except Exception:
            default_lb = None
        sel = 0
        for d in self.p.get_loopback_device_info_generator():
            if d["index"] == default_lb:
                sel = self.dev_combo.count()
            self.dev_combo.addItem("🔊 PC audio · " + d["name"].replace(" [Loopback]", ""), d["index"])
        try:
            wasapi = self.p.get_host_api_info_by_type(pa.paWASAPI)["index"]
        except Exception:
            wasapi = None
        for i in range(self.p.get_device_count()):
            d = self.p.get_device_info_by_index(i)
            if d["maxInputChannels"] > 0 and not d.get("isLoopbackDevice") and d["hostApi"] == wasapi:
                self.dev_combo.addItem("🎤 Mic · " + d["name"], d["index"])
        if keep:
            i = self.dev_combo.findText(keep)
            if i >= 0:
                sel = i
        self.dev_combo.setCurrentIndex(sel)
        self.dev_combo.blockSignals(False)

        self._device_changed()

    def _device_changed(self, *_):
        if self.capture:
            self.capture.stop()
            self.capture = None
        idx = self.dev_combo.currentData()
        if idx is None:
            return
        try:
            dev = self.p.get_device_info_by_index(idx)
            cap = AudioCapture(self.p, dev)
            cap.start()
            self.capture = cap
            if not dev.get("isLoopbackDevice"):
                QtCore.QTimer.singleShot(3000, lambda c=cap: self._check_mic(c))
        except Exception as e:
            self._notify(f"Couldn't open device: {e}", 8000)

    # ---- walls --------------------------------------------------------------
    def _xnorm(self, f):
        x0, x1 = float(self._fx(self.view_lo)), float(self._fx(self.view_hi))
        return (self._fx(f) - x0) / (x1 - x0)

    def _freq_of_xnorm(self, xn):
        x0, x1 = float(self._fx(self.view_lo)), float(self._fx(self.view_hi))
        return float(10 ** (x0 + xn * (x1 - x0)))

    def _set_wall(self, i, f):
        """Move one wall, keeping it inside the view and away from the other wall."""
        lo, hi = self.walls_hz
        if i == 0:
            lo = float(np.clip(f, self.view_lo, hi / MIN_RATIO))
        else:
            hi = float(np.clip(f, lo * MIN_RATIO, self.view_hi))
        self.walls_hz = [lo, hi]
        self._sync_walls()

    def _sync_walls(self):
        lo, hi = self.walls_hz
        self.glow.set_walls(float(self._xnorm(lo)), float(self._xnorm(hi)))

    def _glow_wall_moved(self, i, xn):
        self._set_wall(i, self._freq_of_xnorm(xn))

    def _open_settings(self):
        self.settings_dlg.show()
        self.settings_dlg.raise_()
        self.settings_dlg.activateWindow()

    # ---- axes / threshold ---------------------------------------------------
    def _fx(self, f):
        return np.log10(np.maximum(f, 1e-3))  # frequency axis is logarithmic

    def _update_axes(self, *_):
        self.view_lo, self.view_hi = piano_view_range(self.a4_spin.value())
        self._set_wall(0, self.walls_hz[0])  # re-clamp the walls into the (possibly shifted) view
        self._set_wall(1, self.walls_hz[1])
        self.floor_db = float(self.floor_spin.value())
        self.thr_db = thr = max(self.thr_db, self.floor_db)  # keep the threshold on screen
        for view in (self.glow, self.meter):
            view.set_levels(self.floor_db, thr, self.top_db)
        self._sync_walls()

    def _top_moved(self, db, finished):
        self.top_db = db
        for view in (self.glow, self.meter):
            view.set_levels(self.floor_db, self.thr_db, db)

    def _thr_moved(self, db, finished):
        self.thr_db = db
        if finished:
            self._update_axes()  # legend + views on release
        else:
            for view in (self.glow, self.meter):
                view.set_levels(self.floor_db, db, self.top_db)

    def _apply_tips(self):
        """Show each setting's explanation when hovering its label text."""
        for lab in self.findChildren(QtWidgets.QLabel):
            if lab.text() in TIPS:
                lab.setToolTip(TIPS[lab.text()])

    def _roll_mode_changed(self, freq):
        self.roll_mode_btn.setText("Freq" if freq else "Keys")
        self.roll.set_mode("freq" if freq else "keys")

    def _freq_row(self, db):
        """One row of the frequency roll: spectrum loudness per pixel column, on the spectrum's log axis."""
        w = self.roll.width()
        freqs = self.analyzer.freqs
        key = (w, self.view_lo, self.view_hi, len(freqs), freqs[1])
        if getattr(self, "_froll_key", None) != key:
            lo, hi = max(1, int(self.view_lo / freqs[1])), min(len(freqs), int(self.view_hi / freqs[1]) + 2)
            x0, x1 = np.log10(self.view_lo), np.log10(self.view_hi)
            self._froll_bins = (lo, hi)
            self._froll_px = ((np.log10(freqs[lo:hi]) - x0) / (x1 - x0) * w).astype(np.float32)
            self._froll_key = key
        lo, hi = self._froll_bins
        # Wave Candy style: the whole dB axis becomes brightness (nothing is cut at the threshold),
        # so reverb, overtone tails and noise give a continuous texture under the notes ...
        lev = np.clip((db[lo:hi] - self.floor_db) / (self.top_db - self.floor_db), 0, 1).astype(np.float32)
        v = resample_columns(self._froll_px, lev, w)
        # ... but brightness falls off fast away from each peak's top, so low notes stay narrow even
        # though the analysis window makes them wide on the log axis
        r = self.ROLL_SHARPEN_RADIUS
        pad = np.pad(v, r, mode="edge")
        local_max = np.max(np.lib.stride_tricks.sliding_window_view(pad, 2 * r + 1), axis=1)
        narrow = np.clip(1 - (local_max - v) * 6, 0, 1) ** 2
        row = v ** 2.2 * (self.ROLL_FLOOR + (1 - self.ROLL_FLOOR) * narrow)

        # Below ~C4 the analysis window still makes notes very wide on the log axis (E1 ~100 px), so there
        # each spectral peak is drawn instead as a soft line as wide as the notes around C4-C5 look,
        # on top of the same faint layer; between C4 and G4 the two blend smoothly.
        px = self._froll_px
        lo_x, hi_x = (self._xnorm(midi_to_freq(m, self.a4_spin.value())) * w for m in self.ROLL_BASS_BLEND)
        if lo_x > 0:
            mid = lev[1:-1]
            idx = np.nonzero((mid > lev[:-2]) & (mid >= lev[2:]) & (mid > 0.02))[0] + 1
            idx = idx[px[idx] < hi_x + 3 * self.ROLL_BASS_WIDTH]
            bass = v ** 2.2 * self.ROLL_FLOOR
            if len(idx):
                a, b, c = lev[idx - 1], lev[idx], lev[idx + 1]
                den = a - 2 * b + c
                frac = np.where(den != 0, 0.5 * (a - c) / np.where(den == 0, 1, den), 0.0)
                cx = np.interp(idx + frac, np.arange(len(px)), px)  # sub-bin, sub-pixel centre
                base = np.floor(cx).astype(np.int32)
                sub = (cx - base).astype(np.float32)
                amp = lev[idx] ** 2.2
                sigma = self.ROLL_BASS_WIDTH / 2.355  # FWHM -> gaussian sigma
                reach = int(np.ceil(3 * sigma))
                for off in range(-reach, reach + 2):
                    k = np.exp(-0.5 * ((off + 0.5 - sub) / sigma) ** 2)
                    cols = base + off
                    ok = (cols >= 0) & (cols < w)
                    np.maximum.at(bass, cols[ok], (amp * k)[ok])
            x = np.arange(w)
            blend = np.clip((hi_x - x) / max(1.0, hi_x - lo_x), 0, 1)  # 1 = bass lines, 0 = as above
            row = blend * bass + (1 - blend) * row
        return row

    DECAY_CURVE = 2.5   # >1 gives the slow end (decay 0.8-1) about half of the slider's travel
    DECAY_MAX = 0.98

    def _decay(self):
        """Slider position (0..100) -> per-frame decay factor; the slow end gets most of the travel."""
        pos = self.smooth_slider.value() / 100.0
        return self.DECAY_MAX * (1.0 - (1.0 - pos) ** self.DECAY_CURVE)  # reaches DECAY_MAX exactly at 100

    def _show_decay_value(self, *_):
        """While the Decay slider is dragged or scrolled, show its position (0.00-1.00) under the handle."""
        sl = self.smooth_slider
        self._slider_bubble(sl, f"{sl.value() / sl.maximum():.2f}")

    def _show_opacity_value(self, *_):
        self._slider_bubble(self.opacity_slider, f"{self.opacity_slider.value()}%")

    def _slider_bubble(self, sl, text):
        """Value bubble right under a slider's handle while it is dragged or scrolled."""
        if not (sl.isSliderDown() or sl.underMouse()):
            return
        opt = QtWidgets.QStyleOptionSlider()
        sl.initStyleOption(opt)
        handle = sl.style().subControlRect(QtWidgets.QStyle.CC_Slider, opt, QtWidgets.QStyle.SC_SliderHandle, sl)
        pt = sl.mapTo(self, QtCore.QPoint(handle.center().x(), sl.height() + 2))
        self.decay_bubble.show_at(text, pt.x(), pt.y(), hide_after_ms=0 if sl.isSliderDown() else 900)

    IDLE_ICON = 68  # px: icon glyph shown in the note box while nothing is sounding

    def _show_idle(self):
        """Nothing sounding: show the app's glyph (peak line + dot) in the note box, in the gradient's colours."""
        self._idle = True
        if not hasattr(self, "th"):  # before the first theme: _apply_theme draws it
            return
        if getattr(self, "_idle_pix", None) is None:
            self._idle_pix = self._glyph_pixmap(self.IDLE_ICON)
        self.note_label.setPixmap(self._idle_pix)

    def _glyph_pixmap(self, side):
        """The icon's line-and-dot glyph drawn live, so it follows the chosen gradient and theme."""
        dpr = self.devicePixelRatioF()
        pix = QtGui.QPixmap(int(side * dpr), int(side * dpr))
        pix.setDevicePixelRatio(dpr)
        pix.fill(QtCore.Qt.transparent)
        # geometry of the 1024-unit icon artwork: curve from x 0.16 to 0.84, baseline at y 0.70
        lw, dot_r = 0.045, 0.0332
        extent = 0.68 + lw                       # widest part: the line incl. its round ends
        scale = side / extent
        top = 0.28 - 3.6 * dot_r                 # top of the dot
        bottom = 0.70 + lw / 2
        ox = (side - 0.68 * scale) / 2 - 0.16 * scale
        oy = (side - (bottom - top) * scale) / 2 - top * scale

        def pt(x, y):
            return QtCore.QPointF(ox + x * scale, oy + y * scale)

        xs = np.linspace(0.16, 0.84, 160)
        hs = 0.08 * np.exp(-((xs - 0.34) / 0.05) ** 2) + 0.42 * np.exp(-((xs - 0.56) / 0.045) ** 2)
        path = QtGui.QPainterPath(pt(xs[0], 0.70 - hs[0]))
        for x, h in zip(xs[1:], hs[1:]):
            path.lineTo(pt(x, 0.70 - h))
        dark = QtGui.QColor(self.th["bg"]).lightnessF() < 0.5
        g = QtGui.QLinearGradient(pt(0.16, 0), pt(0.84, 0))
        for pos, t in ((0.0, 0.40), (0.5, 0.70), (1.0, 0.95)):
            g.setColorAt(pos, readable(self.lut_q[int(t * 255)], dark))
        qp = QtGui.QPainter(pix)
        qp.setRenderHint(QtGui.QPainter.Antialiasing, True)
        qp.setPen(QtGui.QPen(QtGui.QBrush(g), lw * scale, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
        qp.drawPath(path)
        peak = int(np.argmax(hs))
        qp.setPen(QtCore.Qt.NoPen)
        qp.setBrush(QtGui.QColor(self.th["fg"]))
        qp.drawEllipse(pt(xs[peak], 0.70 - hs[peak] - 2.6 * dot_r), dot_r * scale, dot_r * scale)
        qp.end()
        return pix

    def _notify(self, text, ms=4000):
        """Short notice at the bottom centre of the window (replaces the old status bar)."""
        c = self.centralWidget()
        self.toast.show_at(text, c.width() / 2, c.height() - 44, hide_after_ms=ms)

    def _tilt_changed(self, v):
        self.tilt_db = v

    def _tilt_curve(self):
        """Per-bin dB offset for the current tilt, pivoting at 1 kHz: a positive tilt raises the lows
        and lowers the highs by `tilt` dB per octave."""
        freqs = self.analyzer.freqs
        key = (len(freqs), freqs[1])
        if getattr(self, "_octaves_key", None) != key:
            self._octaves = np.log2(np.maximum(freqs, 1.0) / 1000.0).astype(np.float32)
            self._octaves_key = key
        return -self.tilt_db * self._octaves

    def _level(self, db):
        """dBFS -> 0..1 position on the colour map (threshold .. colour max)."""
        thr = self.thr_db
        span = max(1.0, self.top_db - thr)
        return np.clip((np.asarray(db) - thr) / span, 0.0, 1.0)

    # ---- per-frame ----------------------------------------------------------
    def _frame(self):
        now = time.perf_counter()
        if self._frame_start is not None:
            # time the loop spent busy since the last frame started (tick + the paints it caused)
            busy = (now - self._frame_start) * 1000 - self._frame_delay
        else:
            busy = 0.0
        self._frame_start = now
        self.roll.frame_scale = float(np.clip((busy + self._frame_delay) / (1000 / 60), 0.25, 4.0))
        self._tick()
        self._frame_delay = max(self.MIN_IDLE_MS, self.FRAME_MS - busy)
        self.timer.start(int(self._frame_delay))

    def _tick(self):
        cap = self.capture
        if cap is None or self.paused:
            return
        cap.pad_silence()
        an = self.analyzer
        an.set_size(self.fft_combo.currentData(), cap.rate)
        x = cap.latest(an.n)
        db = an.spectrum(x, self._decay())
        if self.tilt_db:
            # tilt the whole spectrum so what is drawn is also what is detected
            db = db + self._tilt_curve()

        a4 = self.a4_spin.value()
        thr = self.thr_db
        fmin, fmax = self.walls_hz
        binhz = an.freqs[1]
        i_lo, i_hi = max(1, int(fmin / binhz)), min(len(db) - 1, int(fmax / binhz) + 1)
        peaks = an.peaks(db, i_lo, i_hi, thr, self.rel_spin.value(), self.npeaks_spin.value(), a4,
                         self.drop_harm.isChecked(), self.tune_spin.value())
        lv_peaks = self._level([p[1] for p in peaks]) if peaks else np.array([])

        self._draw_glow(db, peaks, lv_peaks, binhz)
        self.meter.set_level(float(db[i_lo:i_hi].max()) if i_hi > i_lo else -120.0)

        # readout
        now = time.perf_counter()
        if peaks:
            pf, pd, pm = peaks[0]
            cents = (pm - round(pm)) * 100
            self.last_note, self.last_note_time = midi_name(pm), now
            self.note_label.setText(self.last_note)
            self._idle = False
            self._note_style(self._ink(lv_peaks[0]))
            real_db = pd + self.tilt_db * np.log2(pf / 1000.0)  # readout without the tilt
            self.info_label.setText(f"{pf:.1f} Hz   {cents:+.0f}¢   {real_db:.1f} dBFS")
            notes = sorted(zip(peaks, lv_peaks), key=lambda t: t[0][0])
            self.chord_label.setText(self._fit_notes(
                [(midi_name(p[2]), self._ink(l).name()) for p, l in notes]))
        elif self.last_note and now - self.last_note_time < 0.4:
            self._note_style(None)
        else:
            self._show_idle()
            self._note_style(None)
            self.info_label.setText("")
            self.chord_label.setText("")

        # piano: only notes that have sounded for at least the minimum length
        raw = np.full(PIANO_HI - PIANO_LO + 1, -1, np.float32)
        for (pf, pd, pm), lv in zip(peaks, lv_peaks):
            k = int(round(pm)) - PIANO_LO
            if 0 <= k < len(raw):
                raw[k] = max(raw[k], lv)
        levels, backfill = self._gate_notes(raw, now)
        self.roll.push(levels, backfill, self._freq_row(db) if self.roll.mode == "freq" else None)

    def _ink(self, level):
        """Loudness colour for text, kept readable on the theme's background."""
        return readable(self.lut_q[int(level * 255)], QtGui.QColor(self.th["bg"]).lightnessF() < 0.5)

    def _fit_notes(self, notes):
        """Coloured note list that fits the label's width; whatever doesn't fit becomes '…'."""
        fm = self.chord_label.fontMetrics()
        avail = self.chord_label.width()
        sep, ell = "  ", "…"
        parts, used = [], 0
        for i, (name, color) in enumerate(notes):
            piece = (sep if parts else "") + name
            more = fm.horizontalAdvance(sep + ell) if i < len(notes) - 1 else 0
            if used + fm.horizontalAdvance(piece) + more > avail:
                parts.append(f"<span style='color:{self.th['dim']}'>{'&nbsp;&nbsp;' if parts else ''}{ell}</span>")
                break
            used += fm.horizontalAdvance(piece)
            parts.append(f"{'&nbsp;&nbsp;' if parts else ''}<span style='color:{color}'>{name}</span>")
        return "".join(parts)

    GAP = 0.06  # dropouts shorter than this don't end a note

    def _gate_notes(self, raw, now):
        """Hide notes shorter than the minimum length; bridge brief detection dropouts."""
        seen = raw >= 0
        recent = now - self.key_last <= self.GAP
        onset = seen & ~recent
        self.key_since[onset] = now
        self.key_frames[onset] = 0
        self.key_frames[seen] += 1
        self.key_last[seen] = now
        self.key_level[seen] = raw[seen]

        min_d = self.mindur_spin.value() / 1000.0
        if min_d > 0:
            # the FFT window keeps a sound "detected" a bit longer than it really lasts
            min_d += 0.2 * self.analyzer.n / max(1, self.analyzer.rate)
        long_enough = now - self.key_since >= min_d
        shown = seen & long_enough
        held = ~seen & recent & self.key_shown  # keep a confirmed note through a short dropout
        levels = np.where(shown, raw, np.where(held, self.key_level, -1)).astype(np.float32)
        newly = shown & ~self.key_shown & (self.key_frames > 1)
        backfill = [(int(k), int(self.key_frames[k]) - 1) for k in np.nonzero(newly)[0]]
        self.key_shown = shown | held
        return levels, backfill

    def _draw_glow(self, db, peaks, lv_peaks, binhz):
        a4, thr, floor = self.a4_spin.value(), self.thr_db, self.floor_db
        xn = self._xnorm
        lo, hi = max(1, int(self.view_lo / binhz)), min(len(db), int(self.view_hi / binhz) + 2)
        d = db[lo:hi]
        lv = np.where(d > thr, 0.04 + 0.96 * self._level(d), 0).astype(np.float32)
        shape = np.clip((d - floor) / -floor, 0, 1).astype(np.float32)
        guides = [(float(xn(midi_to_freq(m, a4))), midi_name(m)) for m in range(12, 132, 12)
                  if self.view_lo <= midi_to_freq(m, a4) <= self.view_hi]
        pk = [(float(xn(p[0])), midi_name(p[2]), float(l), float(np.clip((p[1] - floor) / -floor, 0, 1)))
              for p, l in zip(peaks, lv_peaks)]
        self.glow.set_data(xn(self.analyzer.freqs[lo:hi]), lv, shape, pk, guides)

    # ---- auto-restart when this file is saved ---------------------------------
    def _watch_source(self):
        if FROZEN:  # the packaged app has no source file to watch
            return
        self._src = os.path.abspath(__file__)
        self._src_mtime = os.path.getmtime(self._src)
        self._reload_timer = QtCore.QTimer(self)
        self._reload_timer.setSingleShot(True)
        self._reload_timer.timeout.connect(self._maybe_restart)
        self._watcher = QtCore.QFileSystemWatcher([self._src], self)
        # editors often write in several steps: wait until the file has settled
        self._watcher.fileChanged.connect(lambda _: self._reload_timer.start(400))

    def _maybe_restart(self):
        if self._src not in self._watcher.files():  # replaced-on-save files drop out of the watch list
            self._watcher.addPath(self._src)
        try:
            mtime = os.path.getmtime(self._src)
            if mtime == self._src_mtime:
                return
            self._src_mtime = mtime
            with open(self._src, encoding="utf-8") as fh:
                compile(fh.read(), self._src, "exec")
        except SyntaxError as e:
            self._notify(f"Not restarted: syntax error on line {e.lineno}: {e.msg}", 10000)
            return
        except OSError:
            return
        self.close()  # saves settings and releases the audio device first
        subprocess.Popen([sys.executable, self._src] + sys.argv[1:], cwd=os.path.dirname(self._src))
        QtWidgets.QApplication.quit()

    def closeEvent(self, ev):
        self._save_settings()
        self.timer.stop()
        self.settings_dlg.close()
        if self.capture:
            self.capture.stop()
        self.p.terminate()
        super().closeEvent(ev)


def main():
    if sys.platform == "win32":
        # own taskbar identity, so Windows shows PitchView's icon instead of Python's
        import ctypes
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("PitchView.PitchView")
        except Exception:
            pass
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    w = MainWindow()
    w.show()
    report = os.environ.get("PITCHVIEW_SELFTEST")
    if report:  # build check: write a short health report, then quit
        QtCore.QTimer.singleShot(3000, lambda: _selftest(w, report))
    sys.exit(app.exec_())


def _selftest(w, path):
    import json
    cap = w.capture
    info = {
        "version": VERSION,
        "frozen": FROZEN,
        "font": QtGui.QFontInfo(w.note_label.font()).family(),
        "svg_images": b"svg" in [bytes(f) for f in QtGui.QImageReader.supportedImageFormats()],
        "devices": w.dev_combo.count(),
        "capture_running": bool(cap and cap.stream and cap.stream.is_active()),
        "icon": not w.windowIcon().isNull(),
        "frames": w.analyzer.smoothed is not None,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=1)
    w.grab().save(os.path.splitext(path)[0] + ".png")
    QtWidgets.QApplication.quit()


if __name__ == "__main__":
    main()
