#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#     "numpy==2.2.6",
#     "tifffile==2025.5.10",
#     "imagecodecs==2025.3.30",
# ]
# ///
"""
evoto_restore.py — Evoto の出力（8bit精度）を、渡す前の 16bit 元データと合成して精度を取り戻す

仕組み
  1. 元データを Evoto の出力プロファイル（Adobe RGB など）に変換し、Evoto と同じ規則で 8bit に丸めた
     「ただ丸めただけの画像」を再現する
  2. Evoto の出力と比べて 2段以上違う画素だけを「レタッチされた画素」とみなし、マスクを作る
     （丸め規則は画像ごとに自動で合わせる）
  3. マスクを少し広げてぼかし、
       レタッチされていない画素 → 元の 16bit データそのまま（色域も元のまま）
       レタッチされた画素       → Evoto の結果
     として元データの色空間（P3 など）・16bit で書き出す

実行には uv を使う（依存ライブラリは上の PEP 723 ブロックと evoto_restore.py.lock で固定）
同じフォルダに gamut_probe.py が必要（プロファイル解析に使う）
使い方:
  uv run evoto_restore.py 元の16bit.tif Evotoの出力.tif [-o 出力.tif] [--mask マスク.tif]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import tifffile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gamut_probe import ICC  # noqa: E402

GRID = np.linspace(0.0, 1.0, 65536)


def read(path):
    with tifffile.TiffFile(path) as tf:
        p = tf.pages[0]
        raw = p.asarray()
        t = p.tags.get("InterColorProfile")
        icc = bytes(t.value) if t is not None else None
        t = p.tags.get("XMP")
        xmp = bytes(t.value) if t is not None else None
        xr, yr, ru = (p.tags.get(k) for k in ("XResolution", "YResolution", "ResolutionUnit"))
        res = None
        if xr is not None and yr is not None:
            res = (xr.value[0] / xr.value[1], yr.value[0] / yr.value[1], int(ru.value) if ru is not None else 2)
    if raw.ndim == 3 and raw.shape[0] in (3, 4) and raw.shape[2] not in (3, 4):
        raw = np.moveaxis(raw, 0, -1)
    if raw.ndim != 3 or raw.shape[2] < 3:
        sys.exit(f"RGB の TIFF ではありません: {path}")
    return raw[..., :3], icc, xmp, res


def norm(raw):
    """uint8 / uint16 -> float64 in [0, 1]（float64 で元データとの完全一致を保証する）"""
    return raw.astype(np.float64) / (65535.0 if raw.dtype == np.uint16 else 255.0)


def encoder(icc):
    """linear -> encoded, per channel (inverse TRC via lookup)."""
    tabs = [icc.trc[c](GRID).astype(np.float64) for c in range(3)]
    return lambda lin: np.stack([np.interp(np.clip(lin[..., c], 0, 1), tabs[c], GRID) for c in range(3)], -1)


def box_blur(m, r):
    """Separable box blur with edge padding (float32)."""
    if r <= 0:
        return m
    out = m
    for ax in (0, 1):
        pad = [(0, 0), (0, 0)]
        pad[ax] = (r + 1, r)
        c = np.cumsum(np.pad(out, pad, mode="edge"), axis=ax, dtype=np.float64)
        hi = [slice(None)] * 2
        lo = [slice(None)] * 2
        hi[ax] = slice(2 * r + 1, None)
        lo[ax] = slice(0, -(2 * r + 1))
        out = ((c[tuple(hi)] - c[tuple(lo)]) / (2 * r + 1)).astype(np.float32)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Evoto の 8bit 出力を元の 16bit と合成して精度を取り戻す")
    ap.add_argument("original", help="Evoto に渡す前の 16bit TIFF（Capture One の書き出し）")
    ap.add_argument("evoto", help="Evoto の出力 TIFF")
    ap.add_argument("-o", "--out", help="出力ファイル（省略時は <evoto>_restored16.tif）")
    ap.add_argument("--mask", help="レタッチ判定マスクを 8bit TIFF で保存（確認用）")
    ap.add_argument("--threshold", type=int, default=2, help="レタッチとみなす差（8bit段, 既定 2）")
    ap.add_argument("--grow", type=int, default=3, help="マスクを広げる量 px（既定 3）")
    ap.add_argument("--feather", type=int, default=6, help="マスク境界のぼかし半径 px（既定 6）")
    ap.add_argument("--chunk", type=int, default=512, help="一度に処理する行数（メモリ節約用）")
    a = ap.parse_args(argv)
    if a.chunk <= 0:
        sys.exit("--chunk は 1 以上を指定してください")

    path = a.out or os.path.splitext(a.evoto)[0] + "_restored16.tif"
    inputs = {os.path.realpath(a.original), os.path.realpath(a.evoto)}
    for label, p in (("出力ファイル", path), ("マスク", a.mask)):
        if p and os.path.realpath(p) in inputs:
            sys.exit(f"{label}が入力ファイルと同じです（上書きすると元データが消えるので中止）: {p}")
    if a.mask and os.path.realpath(a.mask) == os.path.realpath(path):
        sys.exit("出力ファイルとマスクが同じパスです")

    o_raw, o_iccb, xmp, res = read(a.original)
    e_raw, e_iccb, _, _ = read(a.evoto)
    if o_raw.shape != e_raw.shape:
        sys.exit(f"サイズが一致しません: 元 {o_raw.shape[:2]} / Evoto {e_raw.shape[:2]}（トリミング・回転なしで書き出してください）")
    if o_raw.dtype != np.uint16:
        sys.exit("元データが 16bit ではありません")
    if e_raw.dtype not in (np.uint8, np.uint16):
        sys.exit(f"Evoto の出力が未対応の形式です（{e_raw.dtype}）。8bit か 16bit の整数 TIFF で書き出してください")
    if o_iccb is None or e_iccb is None:
        sys.exit("どちらかのファイルに ICC プロファイルが埋め込まれていません")
    oi, ei = ICC(o_iccb), ICC(e_iccb)
    H, W, _ = o_raw.shape
    if e_raw.dtype == np.uint16:
        e8 = ((e_raw.astype(np.int32) + 128) // 257).astype(np.int16)   # 8bit 値（257倍で格納されている想定）
        exact257 = float(np.mean(e_raw[::7, ::7] % 257 == 0))
    else:
        e8, exact257 = e_raw.astype(np.int16), 1.0
    print(f"元データ : {oi.name} / 16bit / {W}×{H}")
    print(f"Evoto    : {ei.name} / {'16bit' if e_raw.dtype == np.uint16 else '8bit'}（8bit精度の画素 {exact257*100:.1f}%）")
    if exact257 < 0.95:
        print("  ! Evoto の出力が 8bit 精度ではないようです。このツールの前提と違うので結果を確認してください。")

    o2e = np.linalg.inv(ei.M) @ oi.M          # original linear -> Evoto linear
    e2o = np.linalg.inv(oi.M) @ ei.M          # Evoto linear  -> original linear
    enc_e, enc_o = encoder(ei), encoder(oi)

    def orig_as_evoto(blk):
        lin = np.stack([oi.trc[c](blk[..., c]) for c in range(3)], -1) @ o2e.T
        return enc_e(lin) * 255.0

    # --- 丸め規則の自動合わせ（間引いた画素で） ----------------------------
    def quantize(v, b):
        return np.clip(np.round(v + b), 0, 255)

    sl = (slice(0, H, 9), slice(0, W, 9))
    v = orig_as_evoto(norm(o_raw[sl]))
    ev = e8[sl]
    best = max(((b, np.mean(np.all(quantize(v, b) == ev, -1))) for b in np.arange(0, 0.55, 0.05)),
               key=lambda t: t[1])
    bias = best[0]
    within1 = np.mean(np.max(np.abs(quantize(v, bias) - ev), -1) <= 1)
    print(f"丸め規則  : round(x + {bias:.2f})  完全一致 {best[1]*100:.1f}% / ±1段以内 {within1*100:.1f}%")

    # --- レタッチ判定 --------------------------------------------------------
    changed = np.zeros((H, W), bool)
    for y in range(0, H, a.chunk):
        q = quantize(orig_as_evoto(norm(o_raw[y:y + a.chunk])), bias)
        changed[y:y + a.chunk] = np.max(np.abs(q - e8[y:y + a.chunk]), -1) >= a.threshold
    raw_frac = changed.mean()
    w = box_blur(changed.astype(np.float32), a.grow) > 0 if a.grow > 0 else changed
    w = box_blur(box_blur(w.astype(np.float32), a.feather), a.feather)
    w = np.clip(w, 0, 1)
    print(f"レタッチ判定: {raw_frac*100:.2f}% の画素 → マスク適用範囲 {np.mean(w > 0)*100:.2f}%（完全に元データ {np.mean(w == 0)*100:.2f}%）")
    if raw_frac > 0.6:
        print("  ! 画像の大半が変化しています（全体補正が入っている可能性）。復元の効果は限定的です。")

    # --- 合成 ---------------------------------------------------------------
    out = np.empty_like(o_raw)
    for y in range(0, H, a.chunk):
        o = norm(o_raw[y:y + a.chunk])
        e = norm(e_raw[y:y + a.chunk])        # Evoto の値をそのまま使う（将来 16bit で来ても精度を捨てない）
        e_lin = np.stack([ei.trc[c](e[..., c]) for c in range(3)], -1) @ e2o.T
        e_in_o = enc_o(e_lin)
        ww = w[y:y + a.chunk, :, None].astype(np.float64)
        res_blk = o * (1 - ww) + e_in_o * ww
        out[y:y + a.chunk] = np.round(np.clip(res_blk, 0, 1) * 65535).astype(np.uint16)

    kw = dict(photometric="rgb", iccprofile=o_iccb, metadata=None, compression=None)
    if res:
        kw.update(resolution=(res[0], res[1]), resolutionunit=res[2])
    extratags = [(700, 1, len(xmp), xmp, True)] if xmp else []
    tifffile.imwrite(path, out, extratags=extratags, software="evoto_restore.py", **kw)

    # --- 自己チェック -------------------------------------------------------
    untouched = w == 0
    diff = np.abs(out.astype(np.int32) - o_raw.astype(np.int32)).max(-1)
    sub = out[::5, ::5].reshape(-1, 3)
    print(f"検証      : 非レタッチ部の元データとの最大差 {int(diff[untouched].max()) if untouched.any() else 0} / 65535")
    print(f"            8bit精度の画素の割合 {np.mean(sub % 257 == 0)*100:.2f}%（元データ同等なら ~0.4%）")
    print(f"出力      : {path}（{oi.name} / 16bit）")
    if a.mask:
        tifffile.imwrite(a.mask, np.round(w * 255).astype(np.uint8), photometric="minisblack", metadata=None)
        print(f"マスク    : {a.mask}")


if __name__ == "__main__":
    main()
