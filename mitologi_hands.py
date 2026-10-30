#!/usr/bin/env python3
"""
Panggung Mitologi Nusantara
===========================
Tokoh mitologi di dalam kubus 3D, dikendalikan dengan tangan lewat webcam.
Tokoh muncul dulu sebagai titik-titik node, lalu perlahan menjadi gambar utuh
bercahaya dengan kilau partikel.

Alur:
  1. Awal: hanya kamera + titik tangan.
  2. Tangan A: tempelkan jempol & telunjuk, lalu buka -> kubus muncul di antara
     kedua jari. Kubus tetap ada selama jari terbuka (lebar jari = ukuran kubus).
     Kubus muncul kosong.
     Tempel lagi jempol & telunjuk -> kubus hilang.
  3. Tangan B: kepalkan -> asap memenuhi kubus.
  4. Tangan B: buka kepalan -> node berkumpul lalu menjadi gambar tokoh (4D).
  5. Kepal-buka lagi dengan tangan B -> tokoh berikutnya.

Tombol: H = titik tangan, F = layar penuh/jendela, I = info/fps, Q/Esc = keluar.
Program terbuka layar penuh; untuk jendela biasa: python mitologi_hands.py --jendela
Mode tanpa kamera (untuk mencoba): python mitologi_hands.py --demo
  lalu tekan spasi untuk "mengepal/membuka".
"""
import argparse
import math
import os
import shutil
import ssl
import subprocess
import sys
import time
import urllib.request

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = ImageDraw = ImageFont = None

import json

from image_nodes import build_character

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tokoh.json")
IMG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gambar")
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
FIG_K = 0.94  # ukuran tokoh relatif terhadap kubus
PINCH_CLOSE = 0.35  # jempol-telunjuk dianggap menempel (relatif ukuran telapak)
PINCH_OPEN = 0.45   # jempol-telunjuk dianggap mulai terbuka (kubus langsung muncul)
PINCH_HOLD = 0.15   # detik minimal jempol-telunjuk menempel sebelum dibuka


def smoothstep(a, b, x):
    t = min(1.0, max(0.0, (x - a) / (b - a)))
    return t * t * (3 - 2 * t)


def make_sprite(d, max_side=520):
    """Gambar tokoh + aura emas + peta kedalaman (untuk efek 4D)."""
    g = d["ghost"]
    H, W = g.shape[:2]
    p = int(max(H, W) * 0.1)
    mask = np.zeros((H + 2 * p, W + 2 * p), np.uint8)
    mask[p:p + H, p:p + W] = g[..., 3]
    col = np.zeros((H + 2 * p, W + 2 * p, 3), np.uint8)
    col[p:p + H, p:p + W] = g[..., :3]
    sc = max_side / max(mask.shape)
    if sc < 1:
        mask = cv2.resize(mask, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        col = cv2.resize(col, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    a_img = cv2.GaussianBlur(mask, (3, 3), 0).astype(np.float32) / 255
    aura = cv2.GaussianBlur(cv2.dilate(mask, np.ones((9, 9), np.uint8)), (0, 0), max(4.0, max(mask.shape) * 0.025))
    a_aura = aura.astype(np.float32) / 255 * 0.8
    gold = np.array([60, 170, 240], np.float32)
    # kedalaman: tengah tubuh paling menonjol (seperti digembungkan)
    dist = cv2.distanceTransform((mask > 128).astype(np.uint8), cv2.DIST_L2, 5)
    depth = np.sqrt(dist / (dist.max() + 1e-6))
    depth = cv2.GaussianBlur(depth, (0, 0), max(4.0, max(mask.shape) * 0.04)).astype(np.float32)
    nx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=5)
    ny = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=5)
    nrm = np.sqrt(nx ** 2 + ny ** 2 + 0.02)
    hgt, wid = mask.shape
    yy, xx = np.mgrid[0:hgt, 0:wid].astype(np.float32)
    x0, x1, yt, yb = d["ghost_box"]
    unit = (x1 - x0) / W / (sc if sc < 1 else 1)
    pad = p * (sc if sc < 1 else 1)
    box = np.array([x0 - pad * unit, x1 + pad * unit, yt + pad * unit, yb - pad * unit])
    return {
        "col": col.astype(np.float32), "a_img": a_img, "a_aura": a_aura, "gold": gold,
        "depth": depth, "nx": nx / nrm, "ny": ny / nrm, "xx": xx, "yy": yy,
    }, box


def smoke_color(hexcol, d):
    """Warna asap (BGR): dari tokoh.json ("#RRGGBB"), atau warna dominan gambar."""
    if hexcol:
        hx = hexcol.lstrip("#")
        return (int(hx[4:6], 16), int(hx[2:4], 16), int(hx[0:2], 16))
    cols = d["cols"]
    hsv = cv2.cvtColor(cols[None].astype(np.uint8), cv2.COLOR_BGR2HSV)[0].astype(float)
    sat = hsv[:, 1] > 80
    c = cols[sat].mean(0) if sat.sum() > 20 else cols.mean(0)
    return tuple(float(v) for v in c / max(c.max(), 1) * 230)


def load_characters():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = json.load(f)
    chars = []
    for c in cfg:
        path = os.path.join(IMG_DIR, c["gambar"])
        if not os.path.exists(path):
            print("  (lewati) gambar tidak ditemukan:", c["gambar"])
            continue
        d = build_character(path, c.get("nama", c["gambar"]), c.get("keterangan", ""),
                            crop=c.get("potong"), mode=c.get("mode", "auto"), threshold=c.get("ambang", 38),
                            cache_dir=CACHE_DIR, margin=tuple(c.get("batas", (0.03, 0.02, 0.03, 0.02))),
                            clear_bright_below=c.get("hapus_terang_bawah"), area=c.get("area"), sure=c.get("pasti"), clear_sky=c.get("hapus_langit", False), fill_holes=c.get("isi_lubang", False))
        d["sprite"], d["sprite_box"] = make_sprite(d)
        d["contour"] = np.nonzero(d["kind"] == 2)[0]
        d["smoke"] = smoke_color(c.get("warna_asap"), d)
        chars.append(d)
        print(f"  tokoh siap: {d['name']}")
    if not chars:
        raise SystemExit("Tidak ada tokoh. Periksa folder 'gambar' dan file tokoh.json.")
    return chars

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "hand_landmarker.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")
WIN = "Panggung Mitologi Nusantara"
PI = math.pi


# ------------------------------------------------------------------ matematika
def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def euler(x, y, z):
    return rot_x(x) @ rot_y(y) @ rot_z(z)


def affine(pos, R, scl):
    M = np.eye(4)
    M[:3, :3] = R * np.asarray(scl, float)
    M[:3, 3] = pos
    return M


def approach(cur, target, rate, dt):
    return cur + (target - cur) * (1 - math.exp(-dt * rate))


CUBE = np.array([[x, y, z] for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)])
CUBE_EDGES = [(a, b) for a in range(8) for b in range(a + 1, 8) if np.sum(np.abs(CUBE[a] - CUBE[b])) == 1]
HAND_CONN = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
             (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18),
             (18, 19), (19, 20), (0, 17)]

SERIF_FONTS = [
    "/System/Library/Fonts/Supplemental/Georgia Bold.ttf",
    "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf",
    "C:/Windows/Fonts/georgiab.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
]
SANS_FONTS = [
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def find_font(cands, size):
    if ImageFont is None:
        return None
    for p in cands:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                pass
    return None


def blit_rgba(frame, rgba, x, y, alpha=1.0):
    h, w = rgba.shape[:2]
    H, W = frame.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    src = rgba[y0 - y:y1 - y, x0 - x:x1 - x]
    a = src[..., 3:4].astype(np.float32) / 255.0 * alpha
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    frame[y0:y1, x0:x1] = (src[..., 2::-1].astype(np.float32) * a + roi * (1 - a)).astype(np.uint8)


# ------------------------------------------------------------------ asap
class Smoke:
    def __init__(self, n=280, seed=3):
        r = np.random.default_rng(seed)
        self.n = n
        self.theta = r.uniform(0, 2 * PI, n)
        self.rad = np.sqrt(r.uniform(0, 1, n)) * 0.44
        self.y = r.uniform(-0.48, 0.48, n)
        self.speed = r.uniform(0.5, 1.4, n) * r.choice([-1, 1], n)
        self.size = r.uniform(0.1, 0.21, n)
        self.phase = r.uniform(0, 2 * PI, n)
        self.shade = r.uniform(0.6, 1.0, n)
        self.tone = r.uniform(0, 1, n)  # campuran warna tokoh & putih supaya asap bergradasi

    def draw(self, app, img, boxM, density, t, color=(215, 222, 230)):
        if density < 0.01:
            return
        ds = 4  # digambar di resolusi 1/4 supaya cepat
        h, w = img.shape[0] // ds, img.shape[1] // ds
        dens = np.zeros((h, w), np.float32)
        tint = np.zeros((h, w), np.float32)
        grow = min(1.0, density * 1.5)  # asap mengembang dari tengah
        th = self.theta + t * self.speed
        rr = self.rad * grow
        yy = (((self.y + 0.5 + t * 0.09 * self.speed) % 1.0) - 0.5) * (0.4 + 0.6 * grow)
        P = np.stack([rr * np.cos(th), yy + 0.05 * np.sin(t * 1.6 + self.phase), rr * np.sin(th)], 1)
        W = P @ boxM[:3, :3].T + boxM[:3, 3]
        P2, z = app.project(W)
        side_px = app.f * np.linalg.norm(boxM[:3, 0]) / app.D
        for i in range(self.n):
            r = int(max(1, self.size[i] * side_px * (0.55 + 0.45 * grow) / ds))
            c = (int(P2[i, 0] / ds), int(P2[i, 1] / ds))
            cv2.circle(dens, c, r, float(self.shade[i]) * 0.55, -1, cv2.LINE_AA)
            cv2.circle(tint, c, r, float(self.shade[i]) * 0.55 * float(self.tone[i]), -1, cv2.LINE_AA)
        sig = max(2.0, side_px / ds * 0.055)
        dens = cv2.GaussianBlur(dens, (0, 0), sig)
        tint = cv2.GaussianBlur(tint, (0, 0), sig)
        mix = np.clip(tint / (dens + 1e-4), 0, 1)  # 0 = putih, 1 = warna tokoh
        dens = cv2.resize(dens, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
        mix = cv2.resize(mix, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)[..., None]
        a = np.clip(dens * density * 2.2, 0, 0.96)[..., None]
        col = np.array(color, np.float32)
        light = col * 0.45 + 255 * 0.55
        smoke = light * (1 - mix) + col * mix
        img[:] = (img.astype(np.float32) * (1 - a) + smoke * a).astype(np.uint8)


# ------------------------------------------------------------------ aplikasi
class App:
    def __init__(self):
        print("Menyiapkan tokoh (pertama kali agak lama, berikutnya cepat)...")
        self.figs = load_characters()
        self.reveal_t0 = 0.0       # waktu tokoh mulai muncul
        self.holder = None         # tangan A (pemegang kubus)
        self.char_on = False       # tokoh sedang ditampilkan?
        r = np.random.default_rng(9)  # node acak yang bertaburan di dalam kubus kosong
        self.idle_base = r.uniform(-0.44, 0.44, (110, 3))
        self.idle_phase = r.uniform(0, 2 * PI, (110, 3))
        self.idle_speed = r.uniform(0.3, 0.9, (110, 3))
        self.idle_col = np.array([[245, 235, 215], [90, 200, 250], [200, 230, 255]])[r.integers(0, 3, 110)].astype(float)
        dd = np.linalg.norm(self.idle_base[:, None] - self.idle_base[None], axis=2)
        self.idle_edges = np.array([(i, j) for i in range(110) for j in np.argsort(dd[i])[1:3] if dd[i, j] < 0.3 and i < j])
        self.holder_seen = -99.0
        self.hide_t0 = 0.0         # waktu tangan mulai mengepal
        self.idx = -1              # belum ada tokoh yang tampil
        self.hands = []
        self.gest_state = {}
        self.last_seen = -99.0
        self.visible = False       # kubus tampil?
        self.armed = False         # sudah mengepal, menunggu tangan dibuka
        self.smoke = 0.0
        self.smoke_target = 0.0
        self.char_alpha = 0.0
        self.box = dict(cx=0.0, cy=0.0, side=3.0, roll=0.0, yaw=0.45, op=0.0)
        self.T = dict(self.box)
        self.cam = False
        self.tracker_ok = False
        self.show_skel = True
        self.show_info = False
        self.smoke_fx = Smoke()
        self.smoke_col = np.array([215.0, 222.0, 230.0])
        self.size = None
        self.D = 10.0
        self.caption_cache = {}
        self.status = ""
        self.demo_fist = False

    def set_size(self, w, h):
        if self.size == (w, h):
            return
        self.size = (w, h)
        self.w, self.h = w, h
        self.f = h * 1.207
        self.cx, self.cy = w / 2, h / 2
        self.f_name = find_font(SERIF_FONTS, max(12, int(30 * h / 720)))
        self.f_sub = find_font(SANS_FONTS, max(10, int(14 * h / 720)))
        self.caption_cache = {}

    def to_world(self, px, py):
        return ((px - self.cx) * self.D / self.f, -(py - self.cy) * self.D / self.f)

    def project(self, P):
        z = np.maximum(self.D - P[..., 2], 0.1)
        return np.stack([self.cx + self.f * P[..., 0] / z, self.cy - self.f * P[..., 1] / z], -1), z

    # --- gerakan tangan
    @staticmethod
    def classify(sc):
        d = lambda a, b: np.linalg.norm(sc[a] - sc[b])
        ext, fold = [], []
        for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18)):
            r = d(tip, 0) / (d(pip, 0) + 1e-6)
            ext.append(r > 1.12)
            fold.append(r < 1.0)
        if fold[0] and fold[1] and fold[2]:
            return "fist"
        if ext[0]:  # telunjuk terbuka (menunjuk atau tangan terbuka) dihitung "buka"
            return "open"
        return "other"

    def process_hands(self, result, now):
        self.hands = []
        lms = (getattr(result, "hand_landmarks", None) if result is not None else None) or []
        hd = (getattr(result, "handedness", None) if result is not None else None) or []
        seen = set()
        events = []  # (key, "fist") / (key, "spread") / (key, "pinch")
        for i, lm in enumerate(lms):
            sc = np.array([[p.x * self.w, p.y * self.h] for p in lm])
            palm = sc[[0, 5, 9, 13, 17]].mean(0)
            w = self.to_world(*palm)
            w0, w9 = self.to_world(*sc[0]), self.to_world(*sc[9])
            size = math.hypot(w0[0] - w9[0], w0[1] - w9[1])
            key = hd[i][0].category_name if i < len(hd) and hd[i] else f"h{i}"
            if key in seen:
                key = f"{key}{i}"
            seen.add(key)
            st = self.gest_state.setdefault(key, {"g": "none", "cand": None, "n": 0,
                                                   "pinch": None, "pcand": None, "pn": 0})
            # bentuk tangan (kepal / buka), harus stabil 3 frame
            raw = self.classify(sc)
            if raw == st["g"]:
                st["n"] = 0
            else:
                if raw == st["cand"]:
                    st["n"] += 1
                else:
                    st["cand"], st["n"] = raw, 1
                if st["n"] >= 2:
                    if raw == "fist":
                        events.append((key, "fist"))
                    st["g"], st["n"] = raw, 0
            # jarak jempol-telunjuk dibanding ukuran telapak
            ratio = np.linalg.norm(sc[4] - sc[8]) / (np.linalg.norm(sc[0] - sc[9]) + 1e-6)
            if raw == "fist":
                praw = "fist"  # kepalan BUKAN jepitan jempol-telunjuk
            elif ratio < PINCH_CLOSE:
                praw = "closed"
            elif ratio > PINCH_OPEN:
                praw = "open"
            else:
                praw = st["pinch"]
            if praw != st["pinch"]:
                if praw == st["pcand"]:
                    st["pn"] += 1
                else:
                    st["pcand"], st["pn"] = praw, 1
                need = 3 if praw == "closed" else 1  # jepitan harus disengaja; membuka langsung terbaca
                if st["pn"] >= need:
                    # kubus hanya muncul bila jempol-telunjuk benar-benar ditempel lalu dibuka
                    if st["pinch"] == "closed" and praw == "open" and now - st.get("closed_t", now) >= PINCH_HOLD:
                        events.append((key, "spread"))
                    if praw == "closed":
                        st["closed_t"] = now
                        events.append((key, "pinch"))
                    st["pinch"], st["pn"] = praw, 0
            else:
                st["pn"] = 0
            self.hands.append(dict(key=key, sc=sc, w=w, size=size, gesture=st["g"], pinch=st["pinch"],
                                   thumb=np.array(self.to_world(*sc[4])), index=np.array(self.to_world(*sc[8])),
                                   palm=np.array(w)))
        for k in list(self.gest_state):
            if k not in seen:
                del self.gest_state[k]
        if self.hands:
            self.last_seen = now
        self.apply_gestures(events, now)

    def apply_gestures(self, events, now):
        holder_here = any(h["key"] == self.holder for h in self.hands)
        if holder_here:
            self.holder_seen = now
        for key, ev in events:
            # Tangan A: tempel jempol-telunjuk lalu buka -> kubus muncul
            if ev == "spread" and (not self.visible or not holder_here):
                if not self.visible:
                    self.visible = True
                    self.smoke = 0.0
                    self.smoke_target = 0.0
                    self.armed = False
                    self.char_on = False  # kubus baru: hanya node acak, tokoh belum muncul
                    self.box["op"] = 0.0
                self.holder = key
                self.holder_seen = now
            # Tangan A: jempol-telunjuk ditempel lagi -> kubus hilang
            elif ev == "pinch" and self.visible and key == self.holder:
                self.visible = False
                self.armed = False
                self.smoke_target = 0.0
            # Tangan B: mengepal -> asap memenuhi kubus
            elif ev == "fist" and self.visible and key != self.holder:
                if not self.armed:
                    self.hide_t0 = now
                self.armed = True
                self.smoke_target = 1.0
        # Tangan B: kepalan dibuka -> tokoh berikutnya muncul (setelah asap penuh)
        b_open = any(h["key"] != self.holder and h["gesture"] == "open" for h in self.hands)
        if self.visible and self.armed and b_open and self.smoke >= 0.7:
            self.idx = (self.idx + 1) % len(self.figs)
            self.reveal_t0 = now
            self.char_on = True
            self.armed = False
            self.smoke_target = 0.0
        # tangan A hilang dari kamera -> kubus hilang
        if self.visible and now - self.holder_seen > 0.8:
            self.visible = False
            self.armed = False
            self.smoke_target = 0.0

    def update_target(self):
        T = self.T
        if not self.visible:
            T["op"] = 0.0
            return
        T["op"] = 1.0
        hold = next((h for h in self.hands if h["key"] == self.holder), None)
        if hold is not None:
            mid = (hold["thumb"] + hold["index"]) / 2
            d = float(np.linalg.norm(hold["thumb"] - hold["index"]))
            side = float(np.clip(d * 1.6, 0.3, 7.5))  # kecil saat baru membuka, membesar sampai jari terbuka penuh
            # arah keluar: dari telapak menuju celah jempol-telunjuk, kubus muncul ke samping
            out = mid - hold["palm"]
            n = float(np.linalg.norm(out))
            out = out / n if n > 1e-6 else np.array([1.0, 0.0])
            c = mid + out * (side * 0.62)
            T.update(cx=float(c[0]), cy=float(c[1]), side=side, roll=0.0, yaw=0.45)
        if self.box["op"] < 0.03:  # baru muncul: langsung di antara jari
            self.box.update({k: T[k] for k in ("cx", "cy", "side", "roll", "yaw")})

    # --- gambar
    def box_matrix(self):
        b = self.box
        return affine((b["cx"], b["cy"], 0), euler(0.2, b["yaw"], b["roll"]), [b["side"]] * 3)

    def draw_edges(self, img, P2, Z, which, op):
        if op <= 0.02:
            return
        ov = img.copy()
        zc = Z.mean()
        for a, b in CUBE_EDGES:
            back = (Z[a] + Z[b]) / 2 > zc
            if back == (which == "back"):
                cv2.line(ov, tuple(np.round(P2[a] * 16).astype(int)), tuple(np.round(P2[b] * 16).astype(int)),
                         (255, 255, 255), 2, cv2.LINE_AA, 4)
        if which == "front":
            for k in range(8):
                cv2.circle(ov, tuple(np.round(P2[k]).astype(int)), max(3, int(self.h / 180)), (255, 255, 255), -1, cv2.LINE_AA)
        a = op * (0.8 if which == "back" else 1.0)
        cv2.addWeighted(ov, a, img, 1 - a, 0, img)

    @staticmethod
    def stage_yaw(t):
        return math.sin(t * 0.5) * 0.5

    def stage_matrix(self, boxM, t):
        return boxM @ affine((0, math.sin(t * 1.4) * 0.02, 0), rot_y(self.stage_yaw(t)), (1, 1, 1))

    def draw_nodes(self, img, fig, stageM, t, alpha, scatter):
        P = fig["nodes"] * FIG_K
        if scatter > 0:
            P = P + fig["scatter"] * scatter * 0.14
        W = P @ stageM[:3, :3].T + stageM[:3, 3]
        P2, _ = self.project(W)
        side_px = self.f * np.linalg.norm(stageM[:3, 0]) / self.D
        lines = np.zeros_like(img)
        E = fig["edges"]
        if len(E):
            pts = np.round(P2 * 16).astype(np.int32)
            ecol = (fig["cols"][E[:, 0]] * 0.45 * alpha).astype(int).tolist()
            for (a, b), c in zip(E, ecol):
                cv2.line(lines, tuple(pts[a]), tuple(pts[b]), c, 1, cv2.LINE_AA, 4)
        tw = 0.65 + 0.35 * np.sin(t * 3 + fig["phase"])  # kerlip
        cols = np.clip(fig["cols"] * (tw * alpha)[:, None], 0, 255).astype(np.uint8)
        dots = np.zeros_like(img)
        x = np.round(P2[:, 0]).astype(int)
        y = np.round(P2[:, 1]).astype(int)
        ok = (x >= 1) & (x < self.w - 1) & (y >= 1) & (y < self.h - 1)
        dots[y[ok], x[ok]] = cols[ok]
        kd = 2 if side_px < 380 else 3
        dots = cv2.dilate(dots, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kd, kd)))
        glow = cv2.GaussianBlur(dots, (0, 0), max(2.0, self.h / 260))
        cv2.add(img, lines, img)
        cv2.add(img, glow, img)
        cv2.add(img, dots, img)

    def sprite_frame(self, spr, yaw, t):
        """Efek 4D: paralaks menurut kedalaman + cahaya yang bergerak + kilatan menyapu."""
        h, w = spr["depth"].shape
        shift = spr["depth"] * (math.sin(yaw) * w * 0.05)
        mx = spr["xx"] - shift
        my = spr["yy"] - spr["depth"] * (math.sin(t * 0.8) * h * 0.008)
        col = cv2.remap(spr["col"], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        a_img = cv2.remap(spr["a_img"], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        nx = cv2.remap(spr["nx"], mx, my, cv2.INTER_LINEAR)
        ny = cv2.remap(spr["ny"], mx, my, cv2.INTER_LINEAR)
        la = t * 0.9 + yaw * 2.0  # arah cahaya ikut berputar
        lx, ly = math.cos(la), math.sin(la) * 0.6 - 0.4
        shade = np.clip(1.0 + 0.3 * (nx * lx + ny * ly), 0.85, 1.25)
        # kilatan cahaya diagonal yang menyapu tubuh setiap ~4 detik
        pos = ((t * 0.25) % 1.0) * (w + h) * 1.4 - (w + h) * 0.2
        band = np.exp(-(((spr["xx"] + spr["yy"]) - pos) / (0.06 * (w + h))) ** 2) * 0.55
        rgb = col * shade[..., None] + 255 * band[..., None] * a_img[..., None]
        a_aura = spr["a_aura"]
        prem = np.clip(rgb, 0, 255) * a_img[..., None] + spr["gold"] * (a_aura * (1 - a_img))[..., None]
        alpha = a_img + a_aura * (1 - a_img)
        return np.dstack([prem, alpha * 255]).astype(np.uint8)

    def draw_sprite(self, img, fig, stageM, alpha, yaw=0.0, t=0.0):
        spr = self.sprite_frame(fig["sprite"], yaw, t)
        x0, x1, yt, yb = fig["sprite_box"] * FIG_K
        Q = np.array([[x0, yt, 0], [x1, yt, 0], [x1, yb, 0], [x0, yb, 0]])
        W = Q @ stageM[:3, :3].T + stageM[:3, 3]
        dst, _ = self.project(W)
        bx0, by0 = np.floor(dst.min(0)).astype(int)
        bx1, by1 = np.ceil(dst.max(0)).astype(int)
        cx0, cy0, cx1, cy1 = max(bx0, 0), max(by0, 0), min(bx1, self.w), min(by1, self.h)
        if cx1 <= cx0 or cy1 <= cy0:
            return
        h, w = spr.shape[:2]
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        M = cv2.getPerspectiveTransform(src, np.float32(dst - [cx0, cy0]))
        warp = cv2.warpPerspective(spr, M, (cx1 - cx0, cy1 - cy0), flags=cv2.INTER_LINEAR)
        a = warp[..., 3:4].astype(np.float32) / 255 * alpha
        roi = img[cy0:cy1, cx0:cx1].astype(np.float32)
        img[cy0:cy1, cx0:cx1] = np.clip(roi * (1 - a) + warp[..., :3].astype(np.float32) * alpha, 0, 255).astype(np.uint8)

    def draw_idle(self, img, boxM, t, alpha):
        """Node-node acak yang melayang & berkerlip di dalam kubus kosong."""
        P = self.idle_base + 0.05 * np.sin(t * self.idle_speed + self.idle_phase)
        W = P @ boxM[:3, :3].T + boxM[:3, 3]
        P2, _ = self.project(W)
        layer = np.zeros_like(img)
        pts = np.round(P2 * 16).astype(np.int32)
        for a, b in self.idle_edges:
            cv2.line(layer, tuple(pts[a]), tuple(pts[b]), (int(120 * alpha),) * 3, 1, cv2.LINE_AA, 4)
        tw = 0.5 + 0.5 * np.sin(t * 2.5 + self.idle_phase[:, 0] * 3)
        dots = np.zeros_like(img)
        x = np.round(P2[:, 0]).astype(int)
        y = np.round(P2[:, 1]).astype(int)
        ok = (x >= 2) & (x < self.w - 2) & (y >= 2) & (y < self.h - 2)
        c = np.clip(self.idle_col * (tw * alpha)[:, None], 0, 255).astype(np.uint8)
        dots[y[ok], x[ok]] = c[ok]
        dots = cv2.dilate(dots, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        glow = cv2.GaussianBlur(dots, (0, 0), max(3.0, self.h / 160))
        cv2.add(img, layer, img)
        cv2.add(img, glow, img)
        cv2.add(img, glow, img)
        cv2.add(img, dots, img)

    def draw_sparkles(self, img, fig, stageM, t, alpha):
        idx = fig["contour"]
        if len(idx) == 0 or alpha < 0.02:
            return
        idx = idx[::2]
        rnd = fig["scatter"][idx]
        P = fig["nodes"][idx] * FIG_K
        drift = (0.5 + 0.5 * np.sin(t * 1.7 + fig["phase"][idx]))[:, None]
        P = P + rnd * 0.018 * drift
        W = P @ stageM[:3, :3].T + stageM[:3, 3]
        P2, _ = self.project(W)
        tw = np.clip(np.sin(t * 4 + fig["phase"][idx] * 3), 0, 1) ** 2  # kilau berkedip
        dots = np.zeros_like(img)
        x = np.round(P2[:, 0]).astype(int)
        y = np.round(P2[:, 1]).astype(int)
        ok = (x >= 1) & (x < self.w - 1) & (y >= 1) & (y < self.h - 1)
        c = (np.array([150, 225, 255]) * (tw * alpha)[:, None]).astype(np.uint8)
        dots[y[ok], x[ok]] = c[ok]
        dots = cv2.dilate(dots, np.ones((2, 2), np.uint8))
        glow = cv2.GaussianBlur(dots, (0, 0), max(2.0, self.h / 300))
        cv2.add(img, dots, img)
        cv2.add(img, glow, img)
        cv2.add(img, glow, img)

    def caption(self, fig):
        key = fig["name"]
        if key not in self.caption_cache:
            if self.f_name is None:
                self.caption_cache[key] = None
            else:
                d0 = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
                w1 = d0.textlength(fig["name"], font=self.f_name)
                w2 = d0.textlength(fig["subtitle"], font=self.f_sub) if self.f_sub else 0
                has_sub = bool(fig["subtitle"]) and self.f_sub is not None
                W, H = int(max(w1, w2) + 20), int(self.h / 720 * (64 if has_sub else 42))
                im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
                d = ImageDraw.Draw(im)
                d.text((W / 2, H * (0.32 if has_sub else 0.5)), fig["name"], font=self.f_name, fill=(240, 185, 70, 255), anchor="mm")
                if has_sub:
                    d.text((W / 2, H * 0.78), fig["subtitle"], font=self.f_sub, fill=(235, 230, 220, 230), anchor="mm")
                self.caption_cache[key] = np.array(im)
        return self.caption_cache[key]

    def render(self, img, now, dt):
        self.set_size(img.shape[1], img.shape[0])
        self.update_target()
        for key in self.box:
            self.box[key] = approach(self.box[key], self.T[key], 25 if key == "op" else 16, dt)
        self.smoke = approach(self.smoke, self.smoke_target, 6.5, dt)
        target_alpha = (1.0 - self.smoke) ** 1.5 if (self.visible and self.char_on and self.idx >= 0) else 0.0
        self.char_alpha = approach(self.char_alpha, target_alpha, 8, dt)

        # titik tangan di belakang kubus supaya tidak menutupi tokoh
        if self.show_skel and self.cam:
            for hnd in self.hands:
                sc = hnd["sc"].astype(int)
                for a, b in HAND_CONN:
                    cv2.line(img, tuple(sc[a]), tuple(sc[b]), (235, 235, 235), 2, cv2.LINE_AA)
                for p in sc:
                    cv2.circle(img, tuple(p), 4, (86, 185, 240), -1, cv2.LINE_AA)
        op = self.box["op"]
        if op > 0.02:
            boxM = self.box_matrix()
            corners = CUBE @ boxM[:3, :3].T + boxM[:3, 3]
            P2, Z = self.project(corners)
            # kaca gelap di dalam kubus supaya tokoh bercahaya kontras
            hull = cv2.convexHull(np.round(P2).astype(np.int32))
            ov = img.copy()
            cv2.fillConvexPoly(ov, hull, (18, 14, 12), cv2.LINE_AA)
            cv2.addWeighted(ov, 0.45 * op, img, 1 - 0.45 * op, 0, img)
            self.draw_edges(img, P2, Z, "back", op)
            if self.idx >= 0 and self.char_alpha > 0.02:
                fig = self.figs[self.idx]
                stageM = self.stage_matrix(boxM, now)
                el = now - self.reveal_t0
                ca = self.char_alpha * op
                # tahap muncul: node berkumpul -> gambar utuh perlahan tampil
                gather = 1.0 - smoothstep(0.0, 0.6, el)
                node_a = 1.0 - smoothstep(0.9, 1.7, el)
                img_a = smoothstep(0.5, 1.4, el)
                if self.armed:  # tangan mengepal: gambar pecah lagi jadi node
                    hh = now - self.hide_t0
                    img_a *= 1.0 - smoothstep(0.0, 0.25, hh)
                    node_a = max(node_a, smoothstep(0.0, 0.25, hh))
                scatter = max(self.smoke, gather)
                if node_a * ca > 0.02:
                    self.draw_nodes(img, fig, stageM, now, node_a * ca, scatter)
                if img_a * ca > 0.02:
                    self.draw_sprite(img, fig, stageM, img_a * ca, self.stage_yaw(now), now)
                    self.draw_sparkles(img, fig, stageM, now, img_a * ca)
            # warna asap = warna tokoh yang akan muncul (saat mengepal) / yang sedang muncul
            nxt = (self.idx + 1) % len(self.figs) if self.armed or self.idx < 0 else self.idx
            k = 1 - math.exp(-dt * 6)
            self.smoke_col += (np.array(self.figs[nxt]["smoke"], float) - self.smoke_col) * k
            self.smoke_fx.draw(self, img, boxM, self.smoke * op, now, tuple(self.smoke_col))
            self.draw_edges(img, P2, Z, "front", op)
            if self.idx >= 0 and self.char_on and self.char_alpha > 0.05:
                cap = self.caption(self.figs[self.idx])
                if cap is not None:
                    bx, top = P2[:, 0].mean(), P2[:, 1].min()
                    y = int(top - cap.shape[0] - 8)
                    if y < 8:  # tidak muat di atas kubus -> taruh di bawah
                        y = int(P2[:, 1].max() + 8)
                    blit_rgba(img, cap, int(bx - cap.shape[1] / 2), y, self.char_alpha * op)

        if self.show_info and self.status:
            cv2.putText(img, self.status, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)


# ------------------------------------------------------------------ MediaPipe
def ensure_model():
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1_000_000:
        return True
    print("File hand_landmarker.task belum ada. Mengunduh (sekitar 7 MB)…")
    try:  # 1) urllib biasa, dengan sertifikat certifi bila ada
        try:
            import certifi
            ctx = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            ctx = ssl.create_default_context()
        with urllib.request.urlopen(MODEL_URL, context=ctx) as r, open(MODEL_PATH, "wb") as f:
            shutil.copyfileobj(r, f)
        print("Model berhasil diunduh.")
        return True
    except Exception as exc:  # noqa: BLE001
        print("  urllib gagal:", exc)
    if shutil.which("curl"):  # 2) pakai curl bawaan Mac
        try:
            subprocess.run(["curl", "-L", "-s", "-o", MODEL_PATH, MODEL_URL], check=True, timeout=120)
            if os.path.getsize(MODEL_PATH) > 1_000_000:
                print("Model berhasil diunduh (curl).")
                return True
        except Exception as exc:  # noqa: BLE001
            print("  curl gagal:", exc)
    print("Gagal mengunduh model. Salin hand_landmarker.task ke folder ini, atau unduh manual dari:")
    print(" ", MODEL_URL)
    return False


def make_landmarker():
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    opts = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp, vision.HandLandmarker.create_from_options(opts)


def screen_size():
    """Ukuran layar (titik). Dipakai supaya gambar kamera mengisi layar penuh."""
    try:
        import tkinter
        root = tkinter.Tk()
        root.withdraw()
        w, h = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        if w > 0 and h > 0:
            return w, h
    except Exception:  # noqa: BLE001
        pass
    return 1440, 900


def fit_cover(frame, aspect):
    """Potong tengah gambar supaya perbandingan lebar:tinggi sama dengan layar."""
    h, w = frame.shape[:2]
    if w / h > aspect:  # gambar lebih lebar dari layar -> potong kiri-kanan
        nw = int(round(h * aspect))
        x0 = (w - nw) // 2
        return frame[:, x0:x0 + nw]
    nh = int(round(w / aspect))
    y0 = (h - nh) // 2
    return frame[y0:y0 + nh]


def message_frame(w, h, lines):
    img = np.full((h, w, 3), (16, 13, 12), np.uint8)
    for i, t in enumerate(lines):
        cv2.putText(img, t, (40, 80 + i * 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (225, 237, 243), 1, cv2.LINE_AA)
    return img


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description="Panggung Mitologi Nusantara")
    ap.add_argument("--camera", type=int, default=0, help="indeks kamera (default 0)")
    ap.add_argument("--demo", action="store_true", help="tanpa kamera; spasi = kepal/buka")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--jendela", action="store_true", help="buka dalam jendela biasa, bukan layar penuh")
    args = ap.parse_args()
    sw, sh = screen_size()
    aspect = sw / sh
    print(f"Layar: {sw}x{sh}")

    app = App()
    cap = landmarker = mp = None
    problem = []
    if not args.demo:
        cap = cv2.VideoCapture(args.camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        if not cap.isOpened():
            cap = None
            problem = ["Kamera tidak bisa dibuka.",
                       "Mac: System Settings > Privacy & Security > Camera,",
                       "aktifkan Visual Studio Code / Terminal, lalu buka ulang VS Code.",
                       "Atau coba kamera lain: python mitologi_hands.py --camera 1"]
        else:
            print("Kamera: OK")
            if ensure_model():
                try:
                    mp, landmarker = make_landmarker()
                    print("Pelacak tangan: OK")
                except Exception as exc:  # noqa: BLE001
                    print("Pelacak tangan GAGAL dimuat:", exc)
            if landmarker is None:
                problem = ["Pelacak tangan tidak aktif - lihat pesan di terminal VS Code."]
    for p in problem:
        print(p)
    app.cam = cap is not None
    app.tracker_ok = landmarker is not None

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    fullscreen = not args.jendela
    if fullscreen:
        cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.resizeWindow(WIN, int(720 * aspect), 720)
    demo_h = 720
    demo_w = int(round(demo_h * aspect))

    t0 = time.perf_counter()
    last = t0
    last_ts = -1
    fps = 0.0
    while True:
        now_abs = time.perf_counter()
        dt = min(0.05, now_abs - last)
        last = now_abs
        now = now_abs - t0
        fps = fps * 0.9 + (1 / max(dt, 1e-3)) * 0.1

        if cap is not None:
            ok, frame = cap.read()
            if not ok:
                print("Gagal membaca kamera.")
                break
            frame = fit_cover(cv2.flip(frame, 1), aspect)  # isi layar penuh, tanpa sisa
        elif args.demo:
            frame = np.full((demo_h, demo_w, 3), (34, 28, 26), np.uint8)
        else:
            frame = message_frame(demo_w, demo_h, problem)

        app.set_size(frame.shape[1], frame.shape[0])
        if landmarker is not None:
            ts = max(int(now * 1000), last_ts + 1)
            last_ts = ts
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
            app.process_hands(res, now)
        elif args.demo:
            app.last_seen = app.holder_seen = now
            app.visible = True
        elif problem and cap is not None:
            y = frame.shape[0] - 30
            cv2.putText(frame, problem[0], (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (138, 155, 255), 2, cv2.LINE_AA)

        app.status = f"{len(app.hands)} tangan | {fps:4.0f} fps"
        app.render(frame, now, dt)
        cv2.imshow(WIN, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("h"):
            app.show_skel = not app.show_skel
        elif key == ord("i"):
            app.show_info = not app.show_info
        elif key == ord("f"):
            fullscreen = not fullscreen
            cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
        elif key == ord(" ") and args.demo:
            app.demo_fist = not app.demo_fist
            if app.demo_fist:
                app.hide_t0, app.armed, app.smoke_target = now, True, 1.0
            elif app.armed:
                app.idx = (app.idx + 1) % len(app.figs)
                app.reveal_t0, app.armed, app.smoke_target, app.char_on = now, False, 0.0, True
        try:
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break

    if cap is not None:
        cap.release()
    if landmarker is not None:
        landmarker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
