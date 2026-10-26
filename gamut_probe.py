#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#     "numpy==2.2.6",
#     "tifffile==2025.5.10",
#     "imagecodecs==2025.3.30",
#     "pillow==12.3.0",
# ]
# ///
"""
gamut_probe.py — アプリの「素通し」色域・ビット深度テスト

  make     : 16bit テストチャート TIFF を生成（ProPhoto 版 / Display P3 版）
  analyze  : 元チャートとアプリから戻ってきた TIFF を比較して HTML レポートを出力

実行には uv を使う（依存ライブラリは上の PEP 723 ブロックと gamut_probe.py.lock で固定）
  uv run gamut_probe.py analyze probe_ProPhoto_16bit.tif Evoto.tif Luminar.tif Affinity.tif -o report.html
"""
from __future__ import annotations

import argparse
import datetime as _dt
import html
import json
import os
import struct
import sys

import numpy as np

try:
    import tifffile
except ImportError:  # pragma: no cover
    sys.exit("tifffile が見つかりません。uv run gamut_probe.py ... の形で実行してください")

VERSION = "1.0"

# ---------------------------------------------------------------------------
# Colorimetry
# ---------------------------------------------------------------------------
PCS_D50 = np.array([0.9642, 1.0, 0.8249])          # ICC PCS white
D65_XY = (0.3127, 0.3290)

BRADFORD = np.array([[0.8951, 0.2664, -0.1614],
                     [-0.7502, 1.7135, 0.0367],
                     [0.0389, -0.0685, 1.0296]])


def xy_to_XYZ(x, y):
    return np.array([x / y, 1.0, (1.0 - x - y) / y])


def bradford(src_white, dst_white):
    s = BRADFORD @ src_white
    d = BRADFORD @ dst_white
    return np.linalg.inv(BRADFORD) @ np.diag(d / s) @ BRADFORD


def rgb_to_xyz_matrix(prim, white_XYZ):
    P = np.column_stack([xy_to_XYZ(*p) for p in prim])
    S = np.linalg.solve(P, white_XYZ)
    return P * S


def d50_matrix(prim, white):
    """RGB(linear) -> XYZ(D50 PCS), i.e. ICC colorants (Bradford-adapted)."""
    if white == "D50":
        return rgb_to_xyz_matrix(prim, PCS_D50)
    w = xy_to_XYZ(*D65_XY)
    return bradford(w, PCS_D50) @ rgb_to_xyz_matrix(prim, w)


GAMUTS = {
    "sRGB":     dict(prim=[(0.640, 0.330), (0.300, 0.600), (0.150, 0.060)], white="D65"),
    "AdobeRGB": dict(prim=[(0.640, 0.330), (0.210, 0.710), (0.150, 0.060)], white="D65"),
    "P3":       dict(prim=[(0.680, 0.320), (0.265, 0.690), (0.150, 0.060)], white="D65"),
    "ProPhoto": dict(prim=[(0.7347, 0.2653), (0.1596, 0.8404), (0.0366, 0.0001)], white="D50"),
}
for _g in GAMUTS.values():
    _g["M"] = d50_matrix(_g["prim"], _g["white"])
    _g["Minv"] = np.linalg.inv(_g["M"])

GAMUT_LABEL = {"sRGB": "sRGB", "AdobeRGB": "Adobe RGB", "P3": "Display P3", "ProPhoto": "ProPhoto"}


def _f(t):
    d = 6 / 29
    return np.where(t > d ** 3, np.cbrt(t), t / (3 * d * d) + 4 / 29)


def _finv(t):
    d = 6 / 29
    return np.where(t > d, t ** 3, 3 * d * d * (t - 4 / 29))


def xyz_to_lab(XYZ):
    XYZ = np.asarray(XYZ, float)
    fx, fy, fz = (_f(XYZ[..., i] / PCS_D50[i]) for i in range(3))
    return np.stack([116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)], -1)


def lab_to_xyz(Lab):
    Lab = np.asarray(Lab, float)
    fy = (Lab[..., 0] + 16) / 116
    fx = fy + Lab[..., 1] / 500
    fz = fy - Lab[..., 2] / 200
    return np.stack([_finv(fx) * PCS_D50[0], _finv(fy) * PCS_D50[1], _finv(fz) * PCS_D50[2]], -1)


def delta_e2000(lab1, lab2):
    L1, a1, b1 = np.moveaxis(np.asarray(lab1, float), -1, 0)
    L2, a2, b2 = np.moveaxis(np.asarray(lab2, float), -1, 0)
    C1 = np.hypot(a1, b1)
    C2 = np.hypot(a2, b2)
    Cm = (C1 + C2) / 2
    G = 0.5 * (1 - np.sqrt(Cm ** 7 / (Cm ** 7 + 25 ** 7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    dLp = L2 - L1
    dCp = C2p - C1p
    dh = h2p - h1p
    dh = np.where(C1p * C2p == 0, 0, np.where(dh > 180, dh - 360, np.where(dh < -180, dh + 360, dh)))
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dh / 2))
    Lpm = (L1 + L2) / 2
    Cpm = (C1p + C2p) / 2
    hsum = h1p + h2p
    hpm = np.where(C1p * C2p == 0, hsum,
                   np.where(np.abs(h1p - h2p) <= 180, hsum / 2,
                            np.where(hsum < 360, (hsum + 360) / 2, (hsum - 360) / 2)))
    T = (1 - 0.17 * np.cos(np.radians(hpm - 30)) + 0.24 * np.cos(np.radians(2 * hpm))
         + 0.32 * np.cos(np.radians(3 * hpm + 6)) - 0.20 * np.cos(np.radians(4 * hpm - 63)))
    dtheta = 30 * np.exp(-(((hpm - 275) / 25) ** 2))
    Rc = 2 * np.sqrt(Cpm ** 7 / (Cpm ** 7 + 25 ** 7))
    Sl = 1 + 0.015 * (Lpm - 50) ** 2 / np.sqrt(20 + (Lpm - 50) ** 2)
    Sc = 1 + 0.045 * Cpm
    Sh = 1 + 0.015 * Cpm * T
    Rt = -np.sin(np.radians(2 * dtheta)) * Rc
    return np.sqrt((dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2 + Rt * (dCp / Sc) * (dHp / Sh))


def inside(XYZ, gamut, tol=0.01):
    rgb = np.asarray(XYZ, float) @ GAMUTS[gamut]["Minv"].T
    return np.all((rgb >= -tol) & (rgb <= 1 + tol), axis=-1)


def srgb_encode(lin):
    lin = np.clip(lin, 0, 1)
    return np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)


def srgb_decode(v):
    v = np.asarray(v, float)
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def xyz_to_srgb_hex(XYZ):
    """Display swatch only (clipped to sRGB)."""
    rgb = srgb_encode(np.clip(np.asarray(XYZ) @ GAMUTS["sRGB"]["Minv"].T, 0, 1))
    r, g, b = (np.round(rgb * 255)).astype(int)
    return f"#{r:02x}{g:02x}{b:02x}"


# ---------------------------------------------------------------------------
# ICC writer (v2.1 matrix/TRC display profile)
# ---------------------------------------------------------------------------
def _s15(v):
    return struct.pack(">i", int(round(v * 65536)))


def _xyz_tag(X):
    return b"XYZ " + b"\0" * 4 + b"".join(_s15(v) for v in X)


def _curv_gamma(g):
    return b"curv" + b"\0" * 4 + struct.pack(">I", 1) + struct.pack(">H", int(round(g * 256)))


def _curv_table(vals):
    v = np.clip(np.round(np.asarray(vals) * 65535), 0, 65535).astype(">u2")
    return b"curv" + b"\0" * 4 + struct.pack(">I", len(v)) + v.tobytes()


def _desc_tag(text):
    a = text.encode("ascii") + b"\0"
    return (b"desc" + b"\0" * 4 + struct.pack(">I", len(a)) + a
            + struct.pack(">II", 0, 0) + struct.pack(">HB", 0, 0) + b"\0" * 67)


def _text_tag(text):
    return b"text" + b"\0" * 4 + text.encode("ascii") + b"\0"


def build_icc(name, M, trc, white_xyz):
    tags = [
        (b"desc", _desc_tag(name)),
        (b"cprt", _text_tag("No copyright, use freely. Generated by gamut_probe.py")),
        (b"wtpt", _xyz_tag(white_xyz)),
        (b"rXYZ", _xyz_tag(M[:, 0])),
        (b"gXYZ", _xyz_tag(M[:, 1])),
        (b"bXYZ", _xyz_tag(M[:, 2])),
    ]
    tags += [(b"rTRC", trc), (b"gTRC", trc), (b"bTRC", trc)]
    n = len(tags)
    offset = 128 + 4 + 12 * n
    table, data = b"", b""
    shared = {}
    for sig, payload in tags:
        if payload in shared:                      # rTRC/gTRC/bTRC share data
            off, size = shared[payload]
        else:
            while (offset + len(data)) % 4:
                data += b"\0"
            off, size = offset + len(data), len(payload)
            data += payload
            shared[payload] = (off, size)
        table += sig + struct.pack(">II", off, size)
    body = struct.pack(">I", n) + table + data
    size = 128 + len(body)
    now = _dt.datetime(2026, 1, 1)
    hdr = (struct.pack(">I", size) + b"lcms" + bytes([2, 0x10, 0, 0]) + b"mntr" + b"RGB " + b"XYZ "
           + struct.pack(">6H", now.year, now.month, now.day, 0, 0, 0)
           + b"acsp" + b"APPL" + b"\0" * 4 + b"\0" * 4 + b"\0" * 4 + b"\0" * 8
           + struct.pack(">I", 0) + b"".join(_s15(v) for v in PCS_D50) + b"\0" * 4 + b"\0" * 16 + b"\0" * 28)
    assert len(hdr) == 128
    return hdr + body


PROPHOTO_GAMMA = 461 / 256          # u8Fixed8 encoding of 1.8, same as Adobe's ProPhoto profile

def make_profiles():
    pp = build_icc("ProPhoto RGB (gamut_probe)", GAMUTS["ProPhoto"]["M"],
                   _curv_gamma(PROPHOTO_GAMMA), PCS_D50)
    p3 = build_icc("Display P3 (gamut_probe)", GAMUTS["P3"]["M"],
                   _curv_table(srgb_decode(np.linspace(0, 1, 1024))), xy_to_XYZ(*D65_XY))
    return {"ProPhoto": pp, "P3": p3}


# ---------------------------------------------------------------------------
# ICC parser (matrix/TRC RGB profiles, v2 and v4)
# ---------------------------------------------------------------------------
class ICC:
    def __init__(self, data: bytes):
        if data is None or len(data) < 132:
            raise ValueError("ICC プロファイルがありません")
        self.data = bytes(data)
        self.version = data[8]
        self.space = data[16:20]
        self.pcs = data[20:24]
        n = struct.unpack(">I", data[128:132])[0]
        self.tags = {}
        for i in range(n):
            sig, off, size = struct.unpack(">4sII", data[132 + 12 * i: 144 + 12 * i])
            self.tags[sig] = data[off: off + size]
        if self.space != b"RGB ":
            raise ValueError(f"RGB 以外のプロファイルです ({self.space!r})")
        need = [b"rXYZ", b"gXYZ", b"bXYZ", b"rTRC", b"gTRC", b"bTRC"]
        if not all(t in self.tags for t in need):
            raise ValueError("LUT 型プロファイルのため未対応（マトリクス/TRC 型のみ対応）")
        self.M = np.column_stack([self._xyz(self.tags[t]) for t in need[:3]])
        self.trc = [self._curve(self.tags[t]) for t in need[3:]]
        self.name = self._desc()

    @staticmethod
    def _xyz(b):
        return np.array(struct.unpack(">3i", b[8:20])) / 65536.0

    @staticmethod
    def _curve(b):
        kind = b[:4]
        if kind == b"curv":
            n = struct.unpack(">I", b[8:12])[0]
            if n == 0:
                return lambda x: x
            if n == 1:
                g = struct.unpack(">H", b[12:14])[0] / 256.0
                return lambda x, g=g: np.clip(x, 0, 1) ** g
            tab = np.frombuffer(b[12:12 + 2 * n], ">u2").astype(float) / 65535
            grid = np.linspace(0, 1, n)
            return lambda x, t=tab, gr=grid: np.interp(x, gr, t)
        if kind == b"para":
            ftype = struct.unpack(">H", b[8:10])[0]
            cnt = {0: 1, 1: 3, 2: 4, 3: 5, 4: 7}[ftype]
            p = [v / 65536 for v in struct.unpack(f">{cnt}i", b[12:12 + 4 * cnt])]
            g = p[0]

            def para(x, ftype=ftype, p=p, g=g):
                x = np.clip(np.asarray(x, float), 0, 1)
                if ftype == 0:
                    return x ** g
                a, bb = p[1], p[2]
                if ftype == 1:
                    return np.where(x >= -bb / a, np.maximum(a * x + bb, 0) ** g, 0)
                if ftype == 2:
                    c = p[3]
                    return np.where(x >= -bb / a, np.maximum(a * x + bb, 0) ** g + c, c)
                c, d = p[3], p[4]
                if ftype == 3:
                    return np.where(x >= d, np.maximum(a * x + bb, 0) ** g, c * x)
                e, f = p[5], p[6]
                return np.where(x >= d, np.maximum(a * x + bb, 0) ** g + e, c * x + f)
            return para
        raise ValueError(f"未対応のトーンカーブ型 {kind!r}")

    def _desc(self):
        b = self.tags.get(b"desc", b"")
        if b[:4] == b"desc":
            n = struct.unpack(">I", b[8:12])[0]
            return b[12:12 + n].split(b"\0")[0].decode("latin-1", "replace")
        if b[:4] == b"mluc":
            cnt = struct.unpack(">I", b[8:12])[0]
            if cnt:
                ln, off = struct.unpack(">II", b[20:28])
                return b[off:off + ln].decode("utf-16-be", "replace")
        return "(名前なし)"

    def to_xyz(self, rgb):
        rgb = np.asarray(rgb, float)
        lin = np.stack([self.trc[i](rgb[..., i]) for i in range(3)], -1)
        return lin @ self.M.T


# ---------------------------------------------------------------------------
# Chart generation
# ---------------------------------------------------------------------------
HUES = list(range(0, 360, 30))
LIGHTNESS = [40, 70]
CAT_ORDER = ["neutral", "in_sRGB", "beyond_sRGB", "Adobe_only", "P3_only", "ProPhoto_only"]
CAT_JA = {
    "neutral": "グレー",
    "in_sRGB": "sRGB内",
    "beyond_sRGB": "sRGB外（Adobe/P3 両方の内側）",
    "Adobe_only": "Adobe RGBだけに入る色（P3外）",
    "P3_only": "P3だけに入る色（Adobe RGB外）",
    "ProPhoto_only": "Adobe/P3 どちらの外（ProPhoto内）",
}

CAT_SHORT = {"neutral": "グレー", "in_sRGB": "sRGB内", "beyond_sRGB": "sRGB外",
             "Adobe_only": "Adobeのみ", "P3_only": "P3のみ", "ProPhoto_only": "PPのみ"}

PATCH_W, PATCH_H, GAP, MARGIN_L, MARGIN_T = 140, 96, 10, 150, 110
W = MARGIN_L + len(HUES) * (PATCH_W + GAP) + 40


def cmax(L, h, gamut):
    lo, hi = 0.0, 250.0
    for _ in range(40):
        m = (lo + hi) / 2
        lab = np.array([L, m * np.cos(np.radians(h)), m * np.sin(np.radians(h))])
        if inside(lab_to_xyz(lab), gamut, tol=0.0):
            lo = m
        else:
            hi = m
    return lo


def classify(XYZ):
    s, a, p, pp = (bool(inside(XYZ, g, 0.002)) for g in ("sRGB", "AdobeRGB", "P3", "ProPhoto"))
    if s:
        return "in_sRGB"
    if a and p:
        return "beyond_sRGB"
    if a:
        return "Adobe_only"
    if p:
        return "P3_only"
    if pp:
        return "ProPhoto_only"
    return "outside_ProPhoto"


def design_patches():
    patches = []
    for L in LIGHTNESS:
        for h in HUES:
            cs = {g: cmax(L, h, g) for g in GAMUTS}
            lo, hi = min(cs["AdobeRGB"], cs["P3"]), max(cs["AdobeRGB"], cs["P3"])
            cands = [0.85 * cs["sRGB"]]
            if lo > cs["sRGB"] * 1.06:
                cands.append((cs["sRGB"] + lo) / 2)
            if hi > lo * 1.06:
                cands.append((lo + hi) / 2)
            if cs["ProPhoto"] > hi * 1.06:
                cands.append(hi + 0.5 * (min(cs["ProPhoto"], hi * 1.8) - hi))
            for C in cands:
                lab = [L, C * np.cos(np.radians(h)), C * np.sin(np.radians(h))]
                patches.append(dict(L=L, h=h, C=round(float(C), 2), lab=[round(float(v), 4) for v in lab],
                                    design_cat=classify(lab_to_xyz(np.array(lab)))))
    return patches


def chart_layout(patches):
    """Rows: per lightness, slot index 0..3 (chroma step). Columns: hue."""
    rects, y = [], MARGIN_T
    for L in LIGHTNESS:
        rowp = [p for p in patches if p["L"] == L]
        slots = max(sum(1 for p in rowp if p["h"] == h) for h in HUES)
        for s in range(slots):
            for ci, h in enumerate(HUES):
                hp = [p for p in rowp if p["h"] == h]
                if s < len(hp):
                    x = MARGIN_L + ci * (PATCH_W + GAP)
                    hp[s]["rect"] = [x, y, PATCH_W, PATCH_H]
                    hp[s]["slot"] = s
            y += PATCH_H + GAP
        y += 20
    neutrals = []
    for i, L in enumerate(range(0, 101, 10)):
        x = MARGIN_L + i * (PATCH_W + GAP) * 12 // 11
        neutrals.append(dict(L=L, h=None, C=0, lab=[float(L), 0.0, 0.0], design_cat="neutral",
                             rect=[x, y, PATCH_W, PATCH_H]))
    y += PATCH_H + GAP + 30
    strips = {
        "full": [MARGIN_L, y, W - MARGIN_L - 40, 60],
        "fine": [MARGIN_L, y + 80, W - MARGIN_L - 40, 60],
    }
    H = y + 80 + 60 + 70
    return patches + neutrals, strips, H


def encode_chart(space, all_patches, strips, H):
    g = GAMUTS[space]
    img = np.full((H, W, 3), 0.0)
    bg_lab = np.array([35.0, 0, 0])

    def enc(XYZ):
        lin = np.asarray(XYZ) @ g["Minv"].T
        lin = np.clip(lin, 0, 1)
        if space == "ProPhoto":
            return lin ** (1 / PROPHOTO_GAMMA)
        return srgb_encode(lin)

    img[:] = enc(lab_to_xyz(bg_lab))
    for p in all_patches:
        x, y, w, h = p["rect"]
        img[y:y + h, x:x + w] = enc(lab_to_xyz(np.array(p["lab"])))
    x, y, w, h = strips["full"]
    img[y:y + h, x:x + w] = np.linspace(0, 1, w)[None, :, None]
    x, y, w, h = strips["fine"]
    img[y:y + h, x:x + w] = np.linspace(0.45, 0.47, w)[None, :, None]
    arr = np.round(img * 65535).astype(np.uint16)
    _draw_labels(arr, space, all_patches, strips)
    return arr


def _draw_labels(arr, space, patches, strips):
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return
    H, Wd = arr.shape[:2]
    mask = Image.new("L", (Wd, H), 0)          # white text
    dark = Image.new("L", (Wd, H), 0)          # black text (on light grays)
    d = ImageDraw.Draw(mask)
    dd = ImageDraw.Draw(dark)
    try:
        big, small = ImageFont.load_default(size=26), ImageFont.load_default(size=13)
    except TypeError:
        big = small = ImageFont.load_default()
    d.text((MARGIN_L, 30), f"GAMUT PROBE v{VERSION}  |  {GAMUT_LABEL[space]} 16-bit  |  pass through the app WITHOUT edits",
           fill=255, font=big)
    for ci, h in enumerate(HUES):
        d.text((MARGIN_L + ci * (PATCH_W + GAP) + 4, MARGIN_T - 20), f"h{h}", fill=255, font=small)
    seen = set()
    for p in patches:
        x, y, w, hh = p["rect"]
        if p["design_cat"] == "neutral":
            (d if p["L"] < 60 else dd).text((x + 4, y + hh - 16), f"L{p['L']}", fill=255, font=small)
            continue
        key = (p["L"], p.get("slot"))
        if key not in seen:
            seen.add(key)
            d.text((14, y + hh // 2 - 7), f"L{p['L']} step{p.get('slot', 0) + 1}", fill=255, font=small)
    d.text((14, strips["full"][1] + 22), "ramp 0-1", fill=255, font=small)
    d.text((14, strips["fine"][1] + 22), "ramp .45-.47", fill=255, font=small)
    arr[np.asarray(mask) > 127] = 65535
    arr[np.asarray(dark) > 127] = 0


def make_charts(outdir):
    os.makedirs(outdir, exist_ok=True)
    profiles = make_profiles()
    patches = design_patches()
    all_p, strips, H = chart_layout(patches)
    written = []
    for space, fname in (("ProPhoto", "probe_ProPhoto_16bit.tif"), ("P3", "probe_DisplayP3_16bit.tif")):
        arr = encode_chart(space, all_p, strips, H)
        meta = dict(tool="gamut_probe", version=VERSION, space=space, width=W, height=H,
                    patches=[{k: p[k] for k in ("L", "h", "C", "lab", "design_cat", "rect")} for p in all_p],
                    strips=strips)
        path = os.path.join(outdir, fname)
        tifffile.imwrite(path, arr, photometric="rgb", iccprofile=profiles[space],
                         description=json.dumps(meta), metadata=None)
        with open(os.path.splitext(path)[0] + ".layout.json", "w") as f:
            json.dump(meta, f)
        written.append(path)
    for space, prof in profiles.items():
        with open(os.path.join(outdir, f"{space}_gamut_probe.icc"), "wb") as f:
            f.write(prof)
    return written


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def read_tiff(path):
    with tifffile.TiffFile(path) as tf:
        page = tf.pages[0]
        arr = page.asarray()
        icc = page.tags.get("InterColorProfile")
        icc = bytes(icc.value) if icc is not None else None
        desc = page.description or ""
        bps = page.bitspersample
    if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[2] not in (3, 4):
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError("RGB の TIFF ではありません")
    alpha = arr.shape[2] > 3
    arr = arr[..., :3]
    if arr.dtype == np.uint8:
        f, bits = arr / 255.0, 8
    elif arr.dtype == np.uint16:
        f, bits = arr / 65535.0, 16
    elif arr.dtype.kind == "f":
        f, bits = arr.astype(float), 32
    else:
        raise ValueError(f"未対応のデータ型 {arr.dtype}")
    return dict(img=f, raw=arr, icc=icc, desc=desc, bits=bits, bps=bps, alpha=alpha)


def load_layout(orig_path, orig):
    side = os.path.splitext(orig_path)[0] + ".layout.json"
    if os.path.exists(side):
        with open(side) as fh:
            return json.load(fh)
    return json.loads(orig["desc"])


def measure(info, layout, scale=(1.0, 1.0)):
    if info["icc"] is None:
        icc, prof_note = None, "ICC プロファイルが埋め込まれていません"
    else:
        icc, prof_note = ICC(info["icc"]), None      # unsupported (LUT) profiles raise ValueError
    sx, sy = scale
    out = []
    for p in layout["patches"]:
        x, y, w, h = p["rect"]
        x0, x1 = int((x + w * 0.3) * sx), int((x + w * 0.7) * sx)
        y0, y1 = int((y + h * 0.3) * sy), int((y + h * 0.7) * sy)
        reg = info["img"][y0:y1, x0:x1].reshape(-1, 3)
        if icc is not None:
            XYZ = icc.to_xyz(reg).mean(0)
        else:   # no / unsupported profile: assume sRGB like most viewers do
            XYZ = (srgb_decode(reg) @ GAMUTS["sRGB"]["M"].T).mean(0)
        out.append(XYZ)
    XYZ = np.array(out)
    return XYZ, (icc.name if icc else None), prof_note


def strip_levels(info, rect, scale=(1.0, 1.0)):
    x, y, w, h = rect
    sx, sy = scale
    row = info["raw"][int((y + h // 2) * sy), int(x * sx):int((x + w) * sx)]
    return int(min(len(np.unique(row[:, c])) for c in range(3)))


def analyze_one(orig_path, orig, layout, ref_XYZ, path):
    r = dict(file=os.path.basename(path), warnings=[])
    try:
        info = read_tiff(path)
    except Exception as e:  # noqa
        r["error"] = f"読み込めませんでした: {e}"
        return r
    H0, W0 = orig["img"].shape[:2]
    H1, W1 = info["img"].shape[:2]
    scale = (1.0, 1.0)
    if (H1, W1) != (H0, W0):
        if abs(W1 / W0 - H1 / H0) < 0.01:
            scale = (W1 / W0, H1 / H0)
            r["warnings"].append(f"サイズが変わっています（{W0}×{H0} → {W1}×{H1}）。比率で補正して測定しました。")
        else:
            r["error"] = f"サイズ/向きが一致しません（{W0}×{H0} → {W1}×{H1}）。トリミングや回転をせずに書き出してください。"
            return r
    try:
        XYZ, pname, pnote = measure(info, layout, scale)
    except ValueError as e:
        r["error"] = f"埋め込みプロファイルを解釈できません: {e}"
        return r
    r["profile"] = pname
    if pnote:
        r["warnings"].append(f"プロファイル: {pnote}（sRGB とみなして測定）")
    r["bits"] = info["bits"]
    if info["alpha"]:
        r["warnings"].append("アルファチャンネル付きで書き出されています（測定には影響なし）")
    lab0, lab1 = xyz_to_lab(ref_XYZ), xyz_to_lab(XYZ)
    de = delta_e2000(lab0, lab1)
    cats = [p["cat"] for p in layout["patches"]]
    r["patches"] = [dict(cat=c, de=float(d), ref=xyz_to_srgb_hex(a), out=xyz_to_srgb_hex(b),
                         lab0=[round(float(v), 1) for v in l0], lab1=[round(float(v), 1) for v in l1],
                         L=p["L"], h=p["h"], C0=round(float(np.hypot(l0[1], l0[2])), 1),
                         C1=round(float(np.hypot(l1[1], l1[2])), 1))
                    for c, d, a, b, l0, l1, p in zip(cats, de, ref_XYZ, XYZ, lab0, lab1, layout["patches"])]
    stats = {}
    for c in CAT_ORDER:
        idx = [i for i, cc in enumerate(cats) if cc == c]
        if idx:
            stats[c] = dict(n=len(idx), mean=float(np.mean(de[idx])), max=float(np.max(de[idx])),
                            ok=int(np.sum(de[idx] < 2.0)))
    r["stats"] = stats

    # --- verdicts -----------------------------------------------------
    base = [i for i, c in enumerate(cats) if c in ("neutral", "in_sRGB")]
    base_mean = float(np.mean(de[base]))
    r["base_mean"] = base_mean
    wide = [i for i, c in enumerate(cats) if c not in ("neutral", "in_sRGB")]
    if base_mean > 3.0:
        r["verdict_profile"] = ("bad", "sRGB内の色やグレーまでズレています。プロファイルの割り当てミス（Assign）か、"
                                       "アプリの自動補正が入っている可能性があります。")
    elif base_mean > 1.0:
        r["verdict_profile"] = ("warn", "sRGB内の色にわずかなズレがあります（変換時の丸め・知覚的圧縮・軽い自動補正の可能性）。")
    else:
        r["verdict_profile"] = ("good", "sRGB内の色とグレーは正しく素通ししています（Convert で正しく処理）。")

    if base_mean > 3.0:
        r["ceiling"] = "判定不可"
        r["verdict_gamut"] = ("info", "sRGB内の色からズレているため、色域の上限は判定できません。"
                                      "先にプロファイル処理（Assign/自動補正）の問題を解決してください。")
    elif not wide:
        r["verdict_gamut"] = ("info", "このチャートにはsRGB外のパッチがありません。")
    elif all(de[i] < 2.0 for i in wide):
        r["ceiling"] = "なし（入力の色域をそのまま保持）"
        r["verdict_gamut"] = ("good", "sRGB外の色もすべて保持されています。色域のボトルネックはありません。")
    else:
        scores = {}
        for g in ("sRGB", "AdobeRGB", "P3", "ProPhoto"):
            in_g = [i for i in wide if inside(ref_XYZ[i], g, 0.002)]
            out_g = [i for i in wide if not inside(ref_XYZ[i], g, 0.002)]
            kept = np.mean([de[i] < 2.0 for i in in_g]) if in_g else 1.0
            contained = np.mean([bool(inside(XYZ[i], g, 0.015)) for i in out_g]) if out_g else 1.0
            scores[g] = (kept + contained) / 2
        best = max(scores, key=scores.get)
        r["ceiling"] = GAMUT_LABEL[best]
        r["ceiling_scores"] = {GAMUT_LABEL[k]: round(float(v), 3) for k, v in scores.items()}
        in_best = [i for i in wide if inside(ref_XYZ[i], best, 0.002)]
        compress = bool(in_best) and float(np.mean(de[in_best])) > 1.5
        method = ("内側の色まで彩度が下がっています → 知覚的に圧縮（perceptual 系の変換）している可能性"
                  if compress or base_mean > 1.0 else
                  "上限の内側の色は保持され、外側だけが境界に寄せられています → クリップ（相対的な色域変換）")
        r["method"] = method
        r["verdict_gamut"] = ("warn", f"実効的な色域の上限は {GAMUT_LABEL[best]} 付近です。{method}。")

    full0, fine0 = strip_levels(orig, layout["strips"]["full"]), strip_levels(orig, layout["strips"]["fine"])
    full1 = strip_levels(info, layout["strips"]["full"], scale)
    fine1 = strip_levels(info, layout["strips"]["fine"], scale)
    r["levels"] = dict(full0=full0, fine0=fine0, full1=full1, fine1=fine1)
    if info["bits"] == 8:
        r["verdict_bits"] = ("bad", "8bit で書き出されています。16bit で書き出せる設定がないか確認してください。")
    elif fine1 <= 24 and fine0 > 200:
        r["verdict_bits"] = ("bad", f"ファイルは16bitですが、微小グラデーションの階調が {fine1} 段しかありません。"
                                    "内部で8bit精度に落ちている可能性が高いです。")
    elif fine1 < fine0 * 0.25:
        r["verdict_bits"] = ("warn", f"16bitですが、微小グラデーションの階調が {fine0} → {fine1} 段に減っています。")
    else:
        r["verdict_bits"] = ("good", f"16bit 精度を保っています（微小グラデーション {fine0} → {fine1} 段）。")
    return r


def analyze(orig_path, paths, out_html):
    orig = read_tiff(orig_path)
    layout = load_layout(orig_path, orig)
    ref_XYZ, ref_name, _ = measure(orig, layout)
    for p, X in zip(layout["patches"], ref_XYZ):
        p["cat"] = "neutral" if p["design_cat"] == "neutral" else classify(X)
    results = [analyze_one(orig_path, orig, layout, ref_XYZ, p) for p in paths]
    html_out = render_html(orig_path, ref_name, layout, results)
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html_out)
    for r in results:
        print_summary(r)
    print(f"\nレポート: {out_html}")
    return results


def print_summary(r):
    print("\n" + "=" * 60)
    print(r["file"])
    if "error" in r:
        print("  ✗", r["error"])
        return
    print(f"  埋め込みプロファイル: {r.get('profile') or 'なし'} / {r['bits']}bit")
    for key in ("verdict_profile", "verdict_gamut", "verdict_bits"):
        lv, msg = r[key]
        print(f"  [{lv}] {msg}")
    for c in CAT_ORDER:
        if c in r["stats"]:
            s = r["stats"][c]
            print(f"    {CAT_JA[c]:<28} n={s['n']:>3}  平均ΔE={s['mean']:5.2f}  最大ΔE={s['max']:5.2f}  保持={s['ok']}/{s['n']}")
    for w in r["warnings"]:
        print("  ! " + w)


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------
def _badge(level):
    return {"good": ("OK", "good"), "warn": ("注意", "warn"), "bad": ("NG", "bad"),
            "info": ("情報", "info")}[level]


def render_html(orig_path, ref_name, layout, results):
    e = html.escape
    space = GAMUT_LABEL.get(layout.get("space"), layout.get("space"))
    sumrows = ""
    for r in results:
        if "error" in r:
            sumrows += f"<tr><td>{e(r['file'])}</td><td colspan=5 class='bad'>{e(r['error'])}</td></tr>"
            continue
        cells = "".join(
            f"<td><span class='b {_badge(r[k][0])[1]}'>{_badge(r[k][0])[0]}</span></td>"
            for k in ("verdict_profile", "verdict_gamut", "verdict_bits"))
        sumrows += (f"<tr><td><b>{e(r['file'])}</b></td><td>{e(r.get('profile') or 'なし')}<br>"
                    f"<span class=mut>{r['bits']}bit</span></td>{cells}"
                    f"<td>{e(r.get('ceiling', '—'))}</td></tr>")
    sections = ""
    for r in results:
        if "error" in r:
            continue
        vb = ""
        for k, t in (("verdict_profile", "プロファイル処理"), ("verdict_gamut", "色域"), ("verdict_bits", "ビット深度")):
            lv, msg = r[k]
            lab, cls = _badge(lv)
            vb += f"<div class='v'><span class='b {cls}'>{lab}</span><b>{t}</b><span>{e(msg)}</span></div>"
        st = "".join(
            f"<tr><td>{e(CAT_JA[c])}</td><td>{s['n']}</td><td>{s['mean']:.2f}</td><td>{s['max']:.2f}</td>"
            f"<td>{s['ok']}/{s['n']}</td></tr>" for c, s in ((c, r['stats'][c]) for c in CAT_ORDER if c in r['stats']))
        grid = ""
        for p in r["patches"]:
            d = p["de"]
            cls = "good" if d < 2 else ("warn" if d < 5 else "bad")
            hue = "" if p["h"] is None else f" h{p['h']}"
            tip = (f"{CAT_JA.get(p['cat'], p['cat'])} | L{p['L']}{hue} | 元 C*={p['C0']} → 戻り C*={p['C1']} | "
                   f"Lab {p['lab0']} → {p['lab1']}")
            grid += (f"<div class='cell {cls}' title='{e(tip)}'><div class='sw'>"
                     f"<i style='background:{p['ref']}'></i><i style='background:{p['out']}'></i></div>"
                     f"<span>{d:.1f}</span><small>{e(CAT_SHORT.get(p['cat'], ''))}</small></div>")
        warn = "".join(f"<li>{e(w)}</li>" for w in r["warnings"])
        lv = r["levels"]
        sc = ""
        if "ceiling_scores" in r:
            sc = "<p class=mut>上限判定スコア（1.0 に近いほどその色域で説明がつく）: " + " / ".join(
                f"{e(k)} {v}" for k, v in r["ceiling_scores"].items()) + "</p>"
        sections += f"""
<section class=card>
  <h2>{e(r['file'])}</h2>
  <p class=mut>埋め込みプロファイル: {e(r.get('profile') or 'なし')} · {r['bits']}bit ·
     階調数（全域 / 微小）: {lv['full0']}/{lv['fine0']} → {lv['full1']}/{lv['fine1']}</p>
  {vb}
  {sc}
  {('<ul class=warns>' + warn + '</ul>') if warn else ''}
  <table><tr><th>カテゴリ</th><th>数</th><th>平均ΔE00</th><th>最大ΔE00</th><th>保持(ΔE&lt;2)</th></tr>{st}</table>
  <p class=mut>パッチ別（左=元 / 右=戻り。sRGB画面用に近似表示なのでsRGB外の色は見た目では差が出ません。数値はΔE00）</p>
  <div class=grid>{grid}</div>
</section>"""
    return f"""<!DOCTYPE html><html lang=ja><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Gamut Probe レポート</title>
<style>
:root{{--bg:#0f1115;--card:#191c23;--line:#2c313b;--tx:#e7e9ec;--mut:#9aa1ab;--good:#4ec97b;--warn:#f5b301;--bad:#ff6b4a;--info:#3fb9c9}}
body{{margin:0;background:var(--bg);color:var(--tx);font-family:-apple-system,"Hiragino Sans","Yu Gothic",Meiryo,sans-serif;line-height:1.6}}
.wrap{{max-width:1080px;margin:0 auto;padding:32px 18px 60px}} h1{{font-size:22px;margin:0 0 4px}}
h2{{font-size:16px;margin:0 0 6px}} .mut{{color:var(--mut);font-size:12.5px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px 20px;margin:18px 0}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin:10px 0}} td,th{{border-bottom:1px solid var(--line);padding:7px 8px;text-align:left}}
th{{color:var(--mut);font-weight:600}}
.b{{display:inline-block;font-size:11px;font-weight:700;padding:1px 8px;border-radius:5px;margin-right:8px}}
.b.good{{background:rgba(78,201,123,.15);color:var(--good)}} .b.warn{{background:rgba(245,179,1,.15);color:var(--warn)}}
.b.bad{{background:rgba(255,107,74,.15);color:var(--bad)}} .b.info{{background:rgba(63,185,201,.15);color:var(--info)}}
td.bad{{color:var(--bad)}}
.v{{display:flex;gap:8px;align-items:baseline;font-size:13.5px;margin:6px 0}} .v b{{white-space:nowrap}}
.warns{{color:var(--warn);font-size:13px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(64px,1fr));gap:5px}}
.cell{{background:#12151a;border:1px solid var(--line);border-radius:6px;padding:4px;text-align:center;font-size:11px}}
.cell .sw{{display:flex;height:22px;border-radius:3px;overflow:hidden}} .cell i{{flex:1}}
.cell span{{display:block;font-weight:700}} .cell small{{display:block;color:var(--mut);font-size:9px;white-space:nowrap;overflow:hidden}}
.cell.good span{{color:var(--good)}} .cell.warn span{{color:var(--warn)}} .cell.bad span{{color:var(--bad)}}
</style></head><body><div class=wrap>
<h1>Gamut Probe レポート</h1>
<p class=mut>元チャート: {e(os.path.basename(orig_path))}（{e(space)} / {e(ref_name or '')}）· 生成 {_dt.datetime.now():%Y-%m-%d %H:%M} · gamut_probe v{VERSION}</p>
<section class=card><h2>サマリー</h2>
<table><tr><th>ファイル</th><th>プロファイル</th><th>プロファイル処理</th><th>色域</th><th>ビット深度</th><th>実効上限</th></tr>{sumrows}</table>
<p class=mut>ΔE00 &lt; 1 はほぼ見分け不能、&lt; 2 は保持とみなす、5 以上は明確な変化。</p></section>
{sections}
</div></body></html>"""


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="アプリの色域・ビット深度素通しテスト")
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make", help="テストチャートを生成")
    m.add_argument("-o", "--outdir", default=".")
    a = sub.add_parser("analyze", help="元チャートと戻りファイルを比較")
    a.add_argument("original")
    a.add_argument("returned", nargs="+")
    a.add_argument("-o", "--out", default="gamut_probe_report.html")
    args = ap.parse_args(argv)
    if args.cmd == "make":
        for p in make_charts(args.outdir):
            print("作成:", p)
    else:
        analyze(args.original, args.returned, args.out)


if __name__ == "__main__":
    main()
