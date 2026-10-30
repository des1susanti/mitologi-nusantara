"""
Ubah gambar menjadi tokoh titik-node bercahaya.

Langkah:
  1. Pisahkan tokoh dari latar (latar polos -> beda warna; latar ramai -> GrabCut).
  2. Ambil titik di garis tepi & detail gambar (Canny) + titik isian.
  3. Warna titik diambil dari gambar asli lalu dicerahkan.
  4. Titik yang berdekatan dihubungkan garis tipis (jaring node).
Hasil disimpan di cache (folder .cache) supaya program cepat saat dibuka lagi.
"""
import hashlib
import json
import os

import cv2
import numpy as np

try:
    from scipy.spatial import cKDTree
except ImportError:  # scipy opsional
    cKDTree = None

MAX_SIDE = 640


def load(path, crop=None):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Gambar tidak bisa dibaca: {path}")
    alpha = None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        alpha = img[..., 3]
        img = img[..., :3]
    if crop:  # [kiri, atas, kanan, bawah] dalam persen
        h, w = img.shape[:2]
        l, t, r, b = crop
        img = img[int(h * t / 100):h - int(h * b / 100), int(w * l / 100):w - int(w * r / 100)]
        if alpha is not None:
            alpha = alpha[int(h * t / 100):h - int(h * b / 100), int(w * l / 100):w - int(w * r / 100)]
    s = MAX_SIDE / max(img.shape[:2])
    if s < 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        if alpha is not None:
            alpha = cv2.resize(alpha, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return img, alpha


def segment(img, alpha=None, mode="auto", threshold=38, margin=(0.03, 0.02, 0.03, 0.02), clear_bright_below=None, area=None, sure=None, clear_sky=False, fill_holes=False):
    m = margin
    """Kembalikan mask tokoh (uint8 0/255)."""
    h, w = img.shape[:2]
    if alpha is not None and (alpha < 250).mean() > 0.05:
        mask = (alpha > 128).astype(np.uint8) * 255
    else:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        b = 4
        border = np.concatenate([lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3),
                                 lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)])
        med = np.median(border, 0)
        spread = np.median(np.linalg.norm(border - med, axis=1))
        uniform = spread < 12 and np.mean(np.linalg.norm(border - med, axis=1) < 25) > 0.7
        if mode == "outline":
            # gambar bergaris tepi (mis. tokoh putih di latar putih): isi semua yang dikelilingi garis
            dist = np.linalg.norm(lab - med, axis=2)
            ink = (dist > threshold).astype(np.uint8) * 255
            ink = cv2.dilate(ink, np.ones((3, 3), np.uint8), iterations=2)
            ff = ink.copy()
            pad = np.zeros((h + 2, w + 2), np.uint8)
            for seed in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)):
                if ff[seed[1], seed[0]] == 0:
                    cv2.floodFill(ff, pad, seed, 128)
            mask = np.where(ff == 128, 0, 255).astype(np.uint8)
            mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
        elif mode == "color" or (mode == "auto" and uniform and not area):
            dist = np.linalg.norm(lab - med, axis=2)
            mask = (dist > threshold).astype(np.uint8) * 255
        elif area:
            # area = poligon kira-kira letak tokoh (pecahan lebar/tinggi); di luar = pasti latar
            mask = np.full((h, w), cv2.GC_BGD, np.uint8)
            poly = np.round(np.array(area, float) * [w - 1, h - 1]).astype(np.int32)
            cv2.fillPoly(mask, [poly], cv2.GC_PR_FGD)
            if sure:  # bagian yang pasti termasuk tokoh (mis. rambut gelap & mahkota); boleh beberapa poligon
                polys = sure if isinstance(sure[0][0], (list, tuple)) else [sure]
                for sp in polys:
                    sp = np.round(np.array(sp, float) * [w - 1, h - 1]).astype(np.int32)
                    cv2.fillPoly(mask, [sp], cv2.GC_FGD)
            bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
            cv2.grabCut(img, mask, None, bgd, fgd, 8, cv2.GC_INIT_WITH_MASK)
            mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        else:
            mask = np.zeros((h, w), np.uint8)
            rect = (int(w * m[0]), int(h * m[1]), int(w * (1 - m[0] - m[2])), int(h * (1 - m[1] - m[3])))
            bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
            cv2.grabCut(img, mask, rect, bgd, fgd, 8, cv2.GC_INIT_WITH_RECT)
            mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    if clear_bright_below is not None:
        # buang cahaya/latar terang (mis. api di belakang kaki) di bagian bawah gambar
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        yy = np.arange(h)[:, None] >= int(h * clear_bright_below)
        bright = (hsv[..., 2] > 140) & (hsv[..., 1] > 90) & yy
        mask[bright] = 0
    if clear_sky:
        # buang warna langit (biru muda / awan putih keabuan) yang ikut terbawa
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hh, ss, vv = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        sky = ((hh >= 85) & (hh <= 115) & (ss < 140) & (vv > 120)) | ((ss < 35) & (vv > 150))
        sky = cv2.morphologyEx(sky.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
        mask[sky] = 0
    k = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    # buang potongan kecil (tulisan, logo, gambar sisipan)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep = 1 + np.where(areas >= areas.max() * 0.12)[0]
        mask = np.isin(lbl, keep).astype(np.uint8) * 255
    # isi lubang kecil di dalam tokoh
    inv = 255 - mask
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(inv, 4)
    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        touches = x == 0 or y == 0 or x + ww >= w or y + hh >= h
        if not touches and (fill_holes or area < mask.size * 0.004):
            mask[lbl == i] = 255
    return mask


def make_nodes(img, mask, n_edge=2600, n_fill=1400, seed=5):
    rng = np.random.default_rng(seed)
    h, w = mask.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 60, 150)
    inner = cv2.erode(mask, np.ones((3, 3), np.uint8))
    edges[inner == 0] = 0
    contour = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    ey, ex = np.nonzero(edges)
    cy, cx = np.nonzero(contour)
    fy, fx = np.nonzero(inner)

    def pick(xs, ys, n):
        if len(xs) == 0:
            return np.zeros((0, 2))
        idx = rng.choice(len(xs), size=min(n, len(xs)), replace=False)
        return np.stack([xs[idx], ys[idx]], 1).astype(float)

    pts = np.vstack([pick(cx, cy, int(n_edge * 0.35)), pick(ex, ey, int(n_edge * 0.65)), pick(fx, fy, n_fill)])
    kind = np.concatenate([np.full(min(int(n_edge * 0.35), len(cx)), 2), np.full(min(int(n_edge * 0.65), len(ex)), 1),
                           np.zeros(min(n_fill, len(fx)))])
    # buang titik yang terlalu rapat
    cell = max(2, int(max(h, w) / 260))
    key = (pts[:, 0] // cell).astype(int) * 100000 + (pts[:, 1] // cell).astype(int)
    _, uniq = np.unique(key, return_index=True)
    pts, kind = pts[uniq], kind[uniq]
    # warna dari gambar, dicerahkan supaya bercahaya
    xi = np.clip(pts[:, 0].astype(int), 0, w - 1)
    yi = np.clip(pts[:, 1].astype(int), 0, h - 1)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    c = hsv[yi, xi]
    c[:, 1] = np.clip(c[:, 1] * 1.25, 0, 255)
    c[:, 2] = np.clip(np.maximum(c[:, 2], 110) * 1.2, 0, 255)
    cols = cv2.cvtColor(c[None].astype(np.uint8), cv2.COLOR_HSV2BGR)[0].astype(float)
    # jaring: tiap titik ke 2 tetangga terdekat
    r = max(h, w) * 0.03
    E = []
    if cKDTree is not None:
        tree = cKDTree(pts)
        d, j = tree.query(pts, k=3)
        for i in range(len(pts)):
            for dd, jj in zip(d[i, 1:], j[i, 1:]):
                if dd < r and i < jj:
                    E.append((i, int(jj)))
    else:
        for i in range(len(pts)):
            dd = np.linalg.norm(pts - pts[i], axis=1)
            dd[i] = 1e9
            for jj in np.argsort(dd)[:2]:
                if dd[jj] < r and i < jj:
                    E.append((i, int(jj)))
    # koordinat ternormalisasi: x kanan, y atas, pusat 0, sisi terpanjang = 1
    ys, xs = np.nonzero(mask)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    s = 1.0 / max(x1 - x0, y1 - y0)
    ccx, ccy = (x0 + x1) / 2, (y0 + y1) / 2
    P = np.stack([(pts[:, 0] - ccx) * s, -(pts[:, 1] - ccy) * s, rng.normal(0, 0.01, len(pts))], 1)
    # gambar "bayangan" (tembus pandang) untuk lapisan isian
    crop_img = img[y0:y1 + 1, x0:x1 + 1]
    crop_mask = mask[y0:y1 + 1, x0:x1 + 1]
    ghost = np.dstack([crop_img, crop_mask])
    ghost_box = np.array([(x0 - ccx) * s, (x1 + 1 - ccx) * s, -(y0 - ccy) * s, -(y1 + 1 - ccy) * s])
    return {
        "nodes": P, "cols": cols, "kind": kind, "edges": np.array(E, int).reshape(-1, 2),
        "ghost": ghost, "ghost_box": ghost_box,
        "phase": rng.uniform(0, 2 * np.pi, len(P)), "scatter": rng.normal(0, 1, (len(P), 3)),
    }


def build_character(path, name, subtitle="", crop=None, mode="auto", threshold=38, cache_dir=None, margin=(0.03, 0.02, 0.03, 0.02), clear_bright_below=None, area=None, sure=None, clear_sky=False, fill_holes=False):
    """Bangun (atau ambil dari cache) data node untuk satu gambar."""
    cfg = json.dumps([crop, mode, threshold, list(margin), clear_bright_below, area, sure, clear_sky, fill_holes, 10])
    with open(path, "rb") as fh:  # kunci cache dari isi gambar + pengaturan
        key = hashlib.md5(fh.read() + cfg.encode()).hexdigest()[:12]
    cache = os.path.join(cache_dir, key + ".npz") if cache_dir else None
    if cache and os.path.exists(cache):
        z = np.load(cache)
        d = {k: z[k] for k in z.files}
    else:
        img, alpha = load(path, crop)
        mask = segment(img, alpha, mode, threshold, margin, clear_bright_below, area, sure, clear_sky, fill_holes)
        d = make_nodes(img, mask)
        if cache:
            os.makedirs(cache_dir, exist_ok=True)
            np.savez_compressed(cache, **d)
    d["name"], d["subtitle"] = name, subtitle
    return d
