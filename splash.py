"""Startup splash screen for Air Canvas.

Run `python splash.py` to save it as thumbnail.png (e.g. for a portfolio),
or call show_splash() to display it before the camera starts.
"""

import sys

import cv2
import numpy as np

W, H = 1280, 720

# BGR colors - kept close to what the app itself draws
BG = (24, 22, 22)
WHITE = (255, 255, 255)
BONE = (224, 224, 224)       # MediaPipe's default connection color
MUTED = (150, 150, 150)
DIM = (95, 95, 95)
GREEN = (0, 255, 0)          # fingertip cursor, same as main.py
RED = (48, 48, 255)          # MediaPipe's default landmark color

# MediaPipe hand connections (same as mp_hands.HAND_CONNECTIONS)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]

# A "pointing" hand (index up, others curled) - the DRAW gesture.
# Normalized coords inside the hand's bounding box, y grows downward.
# Curled fingers: knuckle (PIP) rises above the MCP, then the last two joints
# fold back down onto the palm, drifting slightly toward the thumb.
HAND_POINTS = [
    (0.54, 0.95),                                               # 0 wrist
    (0.41, 0.885), (0.33, 0.80), (0.30, 0.71), (0.36, 0.665),   # 1-4 thumb
    (0.40, 0.58), (0.39, 0.42), (0.385, 0.31), (0.38, 0.21),    # 5-8 index (up)
    (0.50, 0.56), (0.51, 0.49), (0.50, 0.58), (0.49, 0.64),     # 9-12 middle
    (0.60, 0.58), (0.615, 0.515), (0.60, 0.60), (0.585, 0.655), # 13-16 ring
    (0.70, 0.62), (0.715, 0.565), (0.70, 0.635), (0.685, 0.675),# 17-20 pinky
]


def _hand_landmarks(x0, y0, bw, bh):
    return [(int(x0 + px * bw), int(y0 + py * bh)) for px, py in HAND_POINTS]


def _wobble(pts, amount, seed):
    """Add slow, smooth jitter so lines look drawn by a real (shaky) finger."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 1, pts.shape)
    k = 41
    kernel = np.ones(k) / k
    for d in range(2):
        noise[:, d] = np.convolve(np.pad(noise[:, d], k, mode="edge"), kernel, "same")[k:-k]
    noise *= amount / (noise.std() + 1e-6)
    return pts + noise


def _smooth(pts, k=9):
    """Round off sharp corners slightly, like a real air-drawn stroke."""
    kernel = np.ones(k) / k
    out = pts.copy()
    for d in range(2):
        out[:, d] = np.convolve(np.pad(pts[:, d], k, mode="edge"), kernel, "same")[k:-k]
    return out


def _resample(vertices, step=4.0):
    pts = []
    for a, b in zip(vertices[:-1], vertices[1:]):
        a, b = np.array(a, float), np.array(b, float)
        n = max(2, int(np.linalg.norm(b - a) / step))
        pts.extend(a + (b - a) * t for t in np.linspace(0, 1, n, endpoint=False))
    pts.append(np.array(vertices[-1], float))
    return np.array(pts)


def _star_path(tip, radius=125):
    """One-stroke star whose lower-right point lands on the fingertip."""
    tx, ty = tip
    ang = np.radians(-90 + 72 * np.arange(5))
    # place the star so vertex 2 (lower right) is exactly at the fingertip
    cx = tx - radius * np.cos(ang[2])
    cy = ty - radius * np.sin(ang[2])
    v = [(cx + radius * np.cos(a), cy + radius * np.sin(a)) for a in ang]
    order = [2, 4, 1, 3, 0, 2]
    pts = _resample([v[i] for i in order])
    pts = _smooth(_wobble(pts, 5.0, seed=7))
    pts[-1] = tip  # always finish exactly under the cursor
    return pts.astype(np.int32)


def _underline_path(x0, x1, y):
    xs = np.linspace(x0, x1, 120)
    ys = y + 4 * np.sin(np.linspace(0, np.pi, 120)) - np.linspace(0, 6, 120)
    pts = _wobble(np.stack([xs, ys], 1), 1.5, seed=3)
    return pts.astype(np.int32)


def _draw_stroke(img, pts, thickness=7):
    cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, WHITE, thickness, cv2.LINE_AA)


def _draw_hand(img, lm):
    for a, b in HAND_CONNECTIONS:
        cv2.line(img, lm[a], lm[b], BONE, 3, cv2.LINE_AA)
    for i, p in enumerate(lm):
        if i == 8:
            continue
        cv2.circle(img, p, 6, RED, -1, cv2.LINE_AA)
        cv2.circle(img, p, 7, WHITE, 1, cv2.LINE_AA)
    # green fingertip cursor, same as the live app
    cv2.circle(img, lm[8], 10, GREEN, -1, cv2.LINE_AA)


def _text(img, s, org, scale, color, thick=1, font=cv2.FONT_HERSHEY_DUPLEX):
    cv2.putText(img, s, org, font, scale, color, thick, cv2.LINE_AA)


def _draw_text_panel(img, show_prompt):
    x = 90

    _text(img, "Air Canvas", (x - 4, 300), 2.9, WHITE, 5)
    (tw, _), _ = cv2.getTextSize("Air Canvas", cv2.FONT_HERSHEY_DUPLEX, 2.9, 5)
    _draw_stroke(img, _underline_path(x, x + tw, 335), thickness=5)

    _text(img, "draw in the air with your index finger", (x, 400), 0.8, MUTED, 1,
          cv2.FONT_HERSHEY_SIMPLEX)

    rows = [
        ("draw", "index finger up"),
        ("pause", "make a fist"),
        ("clear", "open palm"),
    ]
    y = 490
    for action, gesture in rows:
        _text(img, action, (x, y), 0.75, WHITE, 1)
        _text(img, gesture, (x + 110, y), 0.75, MUTED, 1, cv2.FONT_HERSHEY_SIMPLEX)
        y += 44

    if show_prompt:
        msg = "press any key to start  /  q to quit"
        (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        _text(img, msg, (W - 60 - tw, H - 40), 0.6, DIM, 1, cv2.FONT_HERSHEY_SIMPLEX)


def make_splash(show_prompt=True):
    """Return the 1280x720 BGR splash image."""
    img = np.full((H, W, 3), BG, np.uint8)

    lm = _hand_landmarks(740, 190, 500, 520)
    _draw_stroke(img, _star_path(lm[8]))
    _draw_hand(img, lm)
    _draw_text_panel(img, show_prompt)
    return img


def show_splash(window_name, size=None):
    """Display the splash until a key is pressed. Returns False if Q/Esc was pressed."""
    img = make_splash(show_prompt=True)
    if size is not None:
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    cv2.imshow(window_name, img)
    key = cv2.waitKey(0) & 0xFF
    return key not in (ord('q'), 27)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "thumbnail.png"
    cv2.imwrite(out, make_splash(show_prompt=False))
    print(f"saved {out}")
