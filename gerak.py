"""
Gerakan tokoh dari satu gambar diam.

Setiap tokoh punya daftar "gerak" di tokoh.json. Gambar dilenturkan (warp) setiap frame,
jadi sayap bisa mengepak, api berkobar, rambut berkibar, dsb.

Koordinat yang dipakai (u, v) relatif terhadap kotak tokoh:
  u = 0 kiri ... 1 kanan,   v = 0 atas ... 1 bawah.

Jenis gerak dan pengaturannya (semua boleh dihilangkan, ada nilai bawaannya):
  napas   tubuh mengembang-mengempis dari kaki      kuat, laju
  goyang  tubuh bergoyang ke kiri-kanan dari kaki   kuat, laju
  putar   bagian berputar pada satu titik (sayap,   pusat [u,v], kuat (radian), laju,
          lengan, kepala); cermin=true membuat       cermin, dalam (jarak dari sumbu
          sisi kiri & kanan bergerak berlawanan       tengah yang tetap diam)
  liuk    meliuk seperti ular/ekor                   kuat, laju, gelombang, mulai (v)
  kibar   berkibar seperti rambut/selendang          kuat, laju, gelombang
  api     berkobar seperti nyala api                 kuat, laju
Setiap gerak boleh diberi "area": [u0, v0, u1, v1] agar hanya bagian itu yang bergerak,
dan "fase" (detik) untuk menggeser waktunya.
"""
import math

import numpy as np


def _ss(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _area(U, V, box, f=0.08):
    u0, v0, u1, v1 = box
    return (_ss(u0 - f, u0 + f, U) * (1 - _ss(u1 - f, u1 + f, U)) *
            _ss(v0 - f, v0 + f, V) * (1 - _ss(v1 - f, v1 + f, V))).astype(np.float32)


class Gerak:
    def __init__(self, alpha, cfg):
        self.items = []
        if not cfg:
            return
        h, w = alpha.shape
        ys, xs = np.nonzero(alpha > 0.5)
        if len(xs) == 0:
            return
        self.x0, self.y0 = float(xs.min()), float(ys.min())
        self.bw, self.bh = float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        U = (xx - self.x0) / self.bw
        V = (yy - self.y0) / self.bh
        self.U, self.V = U, V
        for g in cfg:
            g = dict(g)
            j = g.get("jenis")
            W = _area(U, V, g["area"]) if g.get("area") else np.ones_like(U)
            it = {"jenis": j, "kuat": float(g.get("kuat", 0.02)), "laju": float(g.get("laju", 2.0)),
                  "fase": float(g.get("fase", 0.0))}
            if j == "napas":
                it["wy"] = -np.clip(1 - V, 0, 1.1) * W * self.bh
                it["wx"] = (U - 0.5) * np.clip(1 - V, 0, 1.1) * W * self.bw * 0.5
            elif j == "goyang":
                it["wx"] = np.clip(1 - V, 0, 1.1) ** 1.2 * W * self.bw
            elif j == "putar":
                pu, pv = g.get("pusat", [0.5, 0.5])
                it["rx"] = xx - (self.x0 + pu * self.bw)
                it["ry"] = yy - (self.y0 + pv * self.bh)
                if g.get("cermin"):
                    d = float(g.get("dalam", 0.15))
                    off = np.abs(U - pu)
                    W = W * _ss(d, d + 0.12, off)
                    W = W * np.where(U < pu, 1.0, -1.0).astype(np.float32)
                it["w"] = W
            elif j == "liuk":
                m = float(g.get("mulai", 0.0))
                it["w"] = W * _ss(m, m + 0.25, V) * self.bw
                it["k"] = 2 * math.pi * float(g.get("gelombang", 1.0)) * V
            elif j == "kibar":
                k = 2 * math.pi * float(g.get("gelombang", 1.2))
                it["px"] = k * V + 3.0 * U
                it["py"] = k * U
                it["w"] = W
            elif j == "api":
                it["w"] = W * np.clip(1.1 - V, 0, 1.2)
            else:
                continue
            self.items.append(it)

    def disp(self, t):
        """Pergeseran isi gambar (dx, dy) dalam piksel pada waktu t, atau None."""
        if not self.items:
            return None
        dx = np.zeros_like(self.U)
        dy = np.zeros_like(self.U)
        for it in self.items:
            j, k, wt = it["jenis"], it["kuat"], it["laju"] * (t + it["fase"])
            if j == "napas":
                s = k * math.sin(wt)
                dx += it["wx"] * s
                dy += it["wy"] * s
            elif j == "goyang":
                dx += it["wx"] * (k * math.sin(wt))
            elif j == "putar":
                th = it["w"] * (k * math.sin(wt))
                c, s = np.cos(th) - 1, np.sin(th)
                dx += it["rx"] * c - it["ry"] * s
                dy += it["rx"] * s + it["ry"] * c
            elif j == "liuk":
                dx += it["w"] * (k * np.sin(it["k"] - wt))
            elif j == "kibar":
                dx += it["w"] * (k * self.bw) * np.sin(it["px"] - wt)
                dy += it["w"] * (0.3 * k * self.bh) * np.sin(it["py"] - 1.3 * wt)
            elif j == "api":
                U, V, w = self.U, self.V, it["w"]
                dx += w * (k * self.bw) * (np.sin(14 * V - wt + 4 * U) + 0.5 * np.sin(31 * V - 1.7 * wt + 9 * U))
                dy -= w * (0.8 * k * self.bh) * (0.5 + 0.5 * np.sin(9 * U - 1.3 * wt))
        return dx, dy
