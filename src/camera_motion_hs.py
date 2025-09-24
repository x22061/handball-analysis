from __future__ import annotations
r"""
Horn–Schunck法で四隅のコーナーパッチのオプティカルフローを計算し、カメラモーションを推定する

使い方例：python Scripts\camera_motion_hs.py --input_dir "D:\data\shot_videos" --output "D:\data\outputs_camera_motion_hs" --recursive --exts .mp4 .mkv .avi --device auto --mixed_precision --save_overlay_video  --save_field --region_ratio 0.10 --frame_gap 1 --alpha 0.5 --iters 100  --grid_step 40 --scale 20
"""

import os
import csv
import argparse
from typing import Dict, Tuple

import cv2
import numpy as np
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
    # Speed optimizations for inference
    try:
        torch.set_grad_enabled(False)
    except Exception:
        pass
    try:
        import torch.backends.cudnn as cudnn  # type: ignore
        cudnn.benchmark = True
    except Exception:
        pass
except Exception:
    TORCH_AVAILABLE = False

REGION_NAMES = ["tl", "tr", "bl", "br"]


def video_base_name(path: str) -> str:
    base = os.path.splitext(os.path.basename(path))[0]
    return base.replace(' ', '_')


def list_videos(input_dir: str, exts: Tuple[str, ...] | list[str], recursive: bool) -> list[str]:
    # 指定されたディレクトリ内の動画ファイル一覧を取得する
    exts_l = {e.lower() if e.startswith('.') else f'.{e.lower()}' for e in exts}
    vids: list[str] = []
    if recursive:
        for root, _dirs, files in os.walk(input_dir):
            for name in files:
                if os.path.splitext(name)[1].lower() in exts_l:
                    vids.append(os.path.join(root, name))
    else:
        for name in os.listdir(input_dir):
            p = os.path.join(input_dir, name)
            if os.path.isfile(p) and os.path.splitext(name)[1].lower() in exts_l:
                vids.append(p)
    vids.sort()
    return vids


def get_corner_crops(frame_bgr: np.ndarray, ratio: float) -> Dict[str, np.ndarray]:
    # フレーム画像から四隅のコーナーパッチを切り出す
    h, w = frame_bgr.shape[:2]
    cw = max(1, int(w * ratio))
    ch = max(1, int(h * ratio))
    return {
        'tl': frame_bgr[0:ch, 0:cw],
        'tr': frame_bgr[0:ch, w - cw:w],
        'bl': frame_bgr[h - ch:h, 0:cw],
        'br': frame_bgr[h - ch:h, w - cw:w],
    }


# ---- HS with pyramid + warping (borrowed from hs_pyramid pattern) ----
# ピラミッドとワーピングを使ったHorn-Schunck法（hs_pyramidパターンから流用）

def gaussian_pyramid(img: np.ndarray, levels: int, scale: float):
    # 画像のガウシアンピラミッドを生成する
    pyr = [img]
    for _ in range(1, max(1, levels)):
        if abs(scale - 0.5) < 1e-6:
            img = cv2.pyrDown(img)
        else:
            img = cv2.resize(img, (max(1, int(img.shape[1] * scale)), max(1, int(img.shape[0] * scale))), interpolation=cv2.INTER_AREA)
        pyr.append(img)
    return pyr


def warp_image(I: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    # オプティカルフローに基づいて画像をワープする
    h, w = I.shape[:2]
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = (xx + u).astype(np.float32)
    map_y = (yy + v).astype(np.float32)
    return cv2.remap(I, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)


def horn_schunck_step(I1: np.ndarray, I2w: np.ndarray, u: np.ndarray, v: np.ndarray, alpha: float, iters: int):
    # Horn-Schunck法の反復ステップ
    Ix1 = cv2.Sobel(I1, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    Iy1 = cv2.Sobel(I1, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    Ix2 = cv2.Sobel(I2w, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    Iy2 = cv2.Sobel(I2w, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    Ix = 0.5 * (Ix1 + Ix2)
    Iy = 0.5 * (Iy1 + Iy2)
    It = I2w - I1

    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16.0
    alpha2 = alpha * alpha

    for _ in range(max(1, iters)):
        u_bar = cv2.filter2D(u, -1, kernel, borderType=cv2.BORDER_DEFAULT)
        v_bar = cv2.filter2D(v, -1, kernel, borderType=cv2.BORDER_DEFAULT)
        der = Ix * u_bar + Iy * v_bar + It
        denom = alpha2 + Ix * Ix + Iy * Iy
        denom = np.where(denom <= 1e-6, 1e-6, denom)
        u = u_bar - (Ix * der) / denom
        v = v_bar - (Iy * der) / denom
    return u, v


def hs_pyramid_flow(I1: np.ndarray, I2: np.ndarray, alpha: float = 10.0, iters: int = 50, outer_iters: int = 3, pyr_levels: int = 4, pyr_scale: float = 0.5) -> np.ndarray:
    # ピラミッドを用いたHorn-Schunck法によるオプティカルフロー推定
    if I1.ndim == 3:
        I1g = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    else:
        I1g = I1.astype(np.float32)
        if I1g.max() > 1.0:
            I1g = I1g / 255.0
    if I2.ndim == 3:
        I2g = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    else:
        I2g = I2.astype(np.float32)
        if I2g.max() > 1.0:
            I2g = I2g / 255.0

    pyr1 = gaussian_pyramid(I1g, pyr_levels, pyr_scale)
    pyr2 = gaussian_pyramid(I2g, pyr_levels, pyr_scale)

    u = np.zeros_like(pyr1[-1], dtype=np.float32)
    v = np.zeros_like(pyr1[-1], dtype=np.float32)
    for level in range(len(pyr1) - 1, -1, -1):
        I1l = pyr1[level]
        I2l = pyr2[level]
        if u.shape != I1l.shape:
            u = cv2.resize(u, (I1l.shape[1], I1l.shape[0]), interpolation=cv2.INTER_LINEAR) * (1.0 / pyr_scale)
            v = cv2.resize(v, (I1l.shape[1], I1l.shape[0]), interpolation=cv2.INTER_LINEAR) * (1.0 / pyr_scale)
        for _ in range(max(1, outer_iters)):
            I2w = warp_image(I2l, u, v)
            u, v = horn_schunck_step(I1l, I2w, u, v, alpha=alpha, iters=iters)
    flow = np.dstack([u, v]).astype(np.float32)
    flow = np.nan_to_num(flow, nan=0.0, posinf=0.0, neginf=0.0)
    return flow

# ---- Torch (GPU/CPU) implementation of HS pyramid ----
# Torch（GPU/CPU）によるHSピラミッド実装

# Kernel and grid caches to reduce per-call overhead
# カーネルとグリッドのキャッシュで呼び出しごとのオーバーヘッドを削減
_TK_CACHE: dict = {}
_GRID_CACHE: dict = {}

def _get_kernels(device, dtype):
    # デバイスと型に応じたカーネルを取得
    key = (str(device), str(dtype))
    k = _TK_CACHE.get(key)
    if k is None:
        ga = torch.tensor([[1, 2, 1],
                           [2, 4, 2],
                           [1, 2, 1]], device=device, dtype=dtype).view(1, 1, 3, 3) / 16.0
        sx = torch.tensor([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], device=device, dtype=dtype).view(1, 1, 3, 3) / 8.0
        sy = torch.tensor([[-1, -2, -1],
                           [ 0,  0,  0],
                           [ 1,  2,  1]], device=device, dtype=dtype).view(1, 1, 3, 3) / 8.0
        k = {'gauss': ga, 'sobelx': sx, 'sobely': sy}
        _TK_CACHE[key] = k
    return k


def _get_base_grid(device, dtype, H: int, W: int):
    # ベースグリッド（[-1,1]範囲）を取得
    key = (str(device), str(dtype), int(H), int(W))
    g = _GRID_CACHE.get(key)
    if g is None:
        # base grid in [-1, 1]
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype),
            torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype),
            indexing='ij'
        )
        g = torch.stack((xx, yy), dim=-1).unsqueeze(0)  # 1xHxWx2
        _GRID_CACHE[key] = g
    return g

def _autocast(enabled: bool):
    # AMP（自動混合精度）コンテキストを取得
    if not enabled:
        # dummy context
        class _Noop:
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc, tb):
                return False
        return _Noop()
    # Prefer torch.amp.autocast with device type
    try:
        from torch.amp import autocast as amp_autocast  # type: ignore
        return amp_autocast('cuda', enabled=True)
    except Exception:
        try:
            from torch.cuda.amp import autocast as cuda_autocast  # type: ignore
            return cuda_autocast(enabled=True)
        except Exception:
            return _Noop()


def torch_from_bgr_to_gray01(img_bgr: np.ndarray, device: str) -> torch.Tensor:
    # BGR画像をグレースケール化しtorchテンソルへ変換
    if img_bgr.ndim == 3:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_bgr
    g = gray.astype(np.float32)
    if g.max() > 1.0:
        g = g / 255.0
    t = torch.from_numpy(g).to(device=device)
    return t.unsqueeze(0).unsqueeze(0)  # 1x1xHxW


def torch_gaussian_blur_3x3(x: torch.Tensor) -> torch.Tensor:
    # 3x3ガウシアンブラー
    k = _get_kernels(x.device, x.dtype)['gauss']
    return F.conv2d(x, k, padding=1)


def torch_sobel_xy(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Sobelフィルタでx/y方向微分
    # x: Bx1xHxW, returns Ix, Iy scaled similar to CPU (/8.0)
    ks = _get_kernels(x.device, x.dtype)
    Ix = F.conv2d(x, ks['sobelx'], padding=1)
    Iy = F.conv2d(x, ks['sobely'], padding=1)
    return Ix, Iy


def torch_warp(I: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # オプティカルフローに基づいて画像をワープ（torch版）
    # I: Bx1xHxW, u/v: Bx1xHxW, reflection border
    B, _, H, W = I.shape
    base = _get_base_grid(I.device, I.dtype, H, W)  # 1xHxWx2
    # Convert pixel displacement to normalized [-1,1]
    sx = 2.0 * (u / max(1, (W - 1)))
    sy = 2.0 * (v / max(1, (H - 1)))
    grid = base + torch.stack((sx.squeeze(1), sy.squeeze(1)), dim=-1)  # BxHxWx2 (broadcast base)
    return F.grid_sample(I, grid, mode='bilinear', padding_mode='reflection', align_corners=True)


def hs_pyramid_flow_torch_batch(I1_list: list[np.ndarray], I2_list: list[np.ndarray],
                                alpha: float = 10.0, iters: int = 50, outer_iters: int = 3,
                                pyr_levels: int = 4, pyr_scale: float = 0.5,
                                device: str = 'cuda', mixed_precision: bool = False) -> list[np.ndarray]:
    """同じサイズのROIをバッチで処理し、GPUの利用効率を高めるバージョン"""
    assert len(I1_list) == len(I2_list) and len(I1_list) > 0
    # To tensors
    t1 = [torch_from_bgr_to_gray01(im, device) for im in I1_list]
    t2 = [torch_from_bgr_to_gray01(im, device) for im in I2_list]
    I1 = torch.cat(t1, dim=0)  # Bx1xHxW
    I2 = torch.cat(t2, dim=0)

    # Build pyramids (list low->high)
    pyr1 = [I1]
    pyr2 = [I2]
    for _ in range(1, max(1, pyr_levels)):
        h = max(1, int(pyr1[-1].shape[-2] * float(pyr_scale)))
        w = max(1, int(pyr1[-1].shape[-1] * float(pyr_scale)))
        pyr1.append(F.interpolate(pyr1[-1], size=(h, w), mode='area'))
        pyr2.append(F.interpolate(pyr2[-1], size=(h, w), mode='area'))

    u = torch.zeros_like(pyr1[-1])
    v = torch.zeros_like(pyr1[-1])
    ac = _autocast(mixed_precision and device.startswith('cuda'))
    with ac:
        for level in range(len(pyr1) - 1, -1, -1):
            I1l = pyr1[level]
            I2l = pyr2[level]
            if u.shape[-2:] != I1l.shape[-2:]:
                u = F.interpolate(u, size=I1l.shape[-2:], mode='bilinear', align_corners=True) * (1.0 / pyr_scale)
                v = F.interpolate(v, size=I1l.shape[-2:], mode='bilinear', align_corners=True) * (1.0 / pyr_scale)
            # Outer fixed point
            for _ in range(max(1, outer_iters)):
                I2w = torch_warp(I2l, u, v)
                # Derivatives
                Ix1, Iy1 = torch_sobel_xy(I1l)
                Ix2, Iy2 = torch_sobel_xy(I2w)
                Ix = 0.5 * (Ix1 + Ix2)
                Iy = 0.5 * (Iy1 + Iy2)
                It = I2w - I1l
                ks = _get_kernels(I1l.device, I1l.dtype)
                alpha2 = torch.tensor(alpha * alpha, dtype=I1l.dtype, device=I1l.device)
                for _ in range(max(1, iters)):
                    u_bar = F.conv2d(u, ks['gauss'], padding=1)
                    v_bar = F.conv2d(v, ks['gauss'], padding=1)
                    der = Ix * u_bar + Iy * v_bar + It
                    denom = alpha2 + Ix * Ix + Iy * Iy
                    denom = torch.clamp(denom, min=1e-6)
                    u = u_bar - (Ix * der) / denom
                    v = v_bar - (Iy * der) / denom

    # Concatenate on channel then permute -> BxHxWx2
    flow = torch.cat([u, v], dim=1)              # Bx2xHxW
    flow = flow.permute(0, 2, 3, 1).contiguous() # BxHxWx2
    flow_np = flow.detach().float().cpu().numpy()
    flow_np = np.nan_to_num(flow_np, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return [flow_np[i] for i in range(flow_np.shape[0])]


def horn_schunck_step_torch(I1: torch.Tensor, I2w: torch.Tensor, u: torch.Tensor, v: torch.Tensor, alpha: float, iters: int) -> Tuple[torch.Tensor, torch.Tensor]:
    # Horn-Schunck法の反復ステップ（torch版）
    # I1/I2w: Bx1xHxW
    Ix1, Iy1 = torch_sobel_xy(I1)
    Ix2, Iy2 = torch_sobel_xy(I2w)
    Ix = 0.5 * (Ix1 + Ix2)
    Iy = 0.5 * (Iy1 + Iy2)
    It = I2w - I1

    kernel = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=I1.dtype, device=I1.device).view(1, 1, 3, 3) / 16.0
    alpha2 = torch.tensor(alpha * alpha, dtype=I1.dtype, device=I1.device)
    for _ in range(max(1, iters)):
        u_bar = F.conv2d(u, kernel, padding=1)
        v_bar = F.conv2d(v, kernel, padding=1)
        der = Ix * u_bar + Iy * v_bar + It
        denom = alpha2 + Ix * Ix + Iy * Iy
        denom = torch.where(denom <= 1e-6, torch.tensor(1e-6, dtype=denom.dtype, device=denom.device), denom)
        u = u_bar - (Ix * der) / denom
        v = v_bar - (Iy * der) / denom
    return u, v


def hs_pyramid_flow_torch(I1: np.ndarray, I2: np.ndarray, alpha: float = 10.0, iters: int = 50, outer_iters: int = 3, pyr_levels: int = 4, pyr_scale: float = 0.5, device: str = 'cuda', mixed_precision: bool = False) -> np.ndarray:
    # ピラミッドを用いたHorn-Schunck法によるオプティカルフロー推定（torch版）
    # Convert to torch grayscale
    I1t = torch_from_bgr_to_gray01(I1, device)
    I2t = torch_from_bgr_to_gray01(I2, device)

    # Build pyramids (list low->high)
    pyr1 = [I1t]
    pyr2 = [I2t]
    for _ in range(1, max(1, pyr_levels)):
        scale = float(pyr_scale)
        h = max(1, int(pyr1[-1].shape[-2] * scale))
        w = max(1, int(pyr1[-1].shape[-1] * scale))
        pyr1.append(F.interpolate(pyr1[-1], size=(h, w), mode='area'))
        pyr2.append(F.interpolate(pyr2[-1], size=(h, w), mode='area'))

    # operate from smallest to largest
    u = torch.zeros_like(pyr1[-1])
    v = torch.zeros_like(pyr1[-1])
    ac = _autocast(mixed_precision and device.startswith('cuda'))
    with ac:
        for level in range(len(pyr1) - 1, -1, -1):
            I1l = pyr1[level]
            I2l = pyr2[level]
            if u.shape[-2:] != I1l.shape[-2:]:
                u = F.interpolate(u, size=I1l.shape[-2:], mode='bilinear', align_corners=True) * (1.0 / pyr_scale)
                v = F.interpolate(v, size=I1l.shape[-2:], mode='bilinear', align_corners=True) * (1.0 / pyr_scale)
            for _ in range(max(1, outer_iters)):
                I2w = torch_warp(I2l, u, v)
                u, v = horn_schunck_step_torch(I1l, I2w, u, v, alpha=alpha, iters=iters)

    flow = torch.cat([u, v], dim=1)  # 1x2xHxW
    flow = flow.permute(0, 2, 3, 1).contiguous()[0]  # HxWx2
    flow_np = flow.detach().float().cpu().numpy()
    flow_np = np.nan_to_num(flow_np, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return flow_np
# ---- Robust averaging: trimmed mean by magnitude ----
# ロバスト平均化：大きさによるトリム平均

def trimmed_mean_uv_by_magnitude(fx: np.ndarray, fy: np.ndarray, low_frac: float = 0.2, high_frac: float = 0.2) -> Tuple[float, float]:
    """(u,v)の大きさで中央部分のみを使って平均を計算する
    最小の`low_frac`と最大の`high_frac`を除外し、残りを平均する
    サンプル数が少ない場合は安全にフォールバック
    """
    # Flatten
    fx_flat = fx.reshape(-1).astype(np.float32)
    fy_flat = fy.reshape(-1).astype(np.float32)
    mag = np.hypot(fx_flat, fy_flat)
    # Keep finite
    finite = np.isfinite(mag)
    fx_f = fx_flat[finite]
    fy_f = fy_flat[finite]
    mag_f = mag[finite]
    N = int(mag_f.size)
    if N == 0:
        return float(np.mean(fx_flat) if fx_flat.size else 0.0), float(np.mean(fy_flat) if fy_flat.size else 0.0)
    # Clamp fractions
    low = max(0.0, min(0.49, float(low_frac)))
    high = max(0.0, min(0.49, float(high_frac)))
    if (low + high) >= 0.99:
        # ensure at least 1% remains
        high = max(0.0, 0.99 - low)
    # Sort by magnitude
    order = np.argsort(mag_f)
    k0 = int(np.floor(low * N))
    k1 = int(np.ceil((1.0 - high) * N))
    # Ensure non-empty slice
    k1 = max(k1, k0 + 1)
    sel = order[k0:k1]
    mu_x = float(np.mean(fx_f[sel]))
    mu_y = float(np.mean(fy_f[sel]))
    return mu_x, mu_y



# ---- Bilinear interpolation of corner vectors over full frame ----
# 四隅ベクトルの全フレームへのバイリニア補間

def bilinear_field(h: int, w: int, tl: Tuple[float, float], tr: Tuple[float, float], bl: Tuple[float, float], br: Tuple[float, float]) -> np.ndarray:
    """4つのコーナーベクトルをバイリニア補間して密なフィールド（H x W x 2）を返す
    座標は幅方向s[0,1]、高さ方向t[0,1]で正規化
    V(s,t) = (1-s)(1-t)TL + s(1-t)TR + (1-s)t BL + s t BR
    """
    s = (np.arange(w, dtype=np.float32)[None, :] / max(1, (w - 1)))
    t = (np.arange(h, dtype=np.float32)[:, None] / max(1, (h - 1)))
    w_tl = (1 - s) * (1 - t)
    w_tr = s * (1 - t)
    w_bl = (1 - s) * t
    w_br = s * t
    tl_vec = np.array(tl, dtype=np.float32)
    tr_vec = np.array(tr, dtype=np.float32)
    bl_vec = np.array(bl, dtype=np.float32)
    br_vec = np.array(br, dtype=np.float32)
    u = w_tl * tl_vec[0] + w_tr * tr_vec[0] + w_bl * bl_vec[0] + w_br * br_vec[0]
    v = w_tl * tl_vec[1] + w_tr * tr_vec[1] + w_bl * bl_vec[1] + w_br * br_vec[1]
    return np.dstack([u, v]).astype(np.float32)


def draw_grid_arrows(bgr: np.ndarray, field: np.ndarray, step: int = 40, scale: float = 20.0, min_pixels: float = 0.0) -> np.ndarray:
    # グリッド上に矢印を描画する
    out = bgr.copy()
    h, w = bgr.shape[:2]
    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):
            u, v = field[y, x]
            dx = u * scale
            dy = v * scale
            if min_pixels > 0:
                mag = float(np.hypot(dx, dy))
                if mag > 0 and mag < min_pixels:
                    f = min_pixels / mag
                    dx *= f
                    dy *= f
            ox, oy = x, y
            ex = int(round(ox + dx))
            ey = int(round(oy + dy))
            cv2.arrowedLine(out, (ox, oy), (ex, ey), (0, 255, 255), 2, tipLength=0.3)
            cv2.circle(out, (ox, oy), 2, (0, 255, 255), -1)
    return out


def process(args: argparse.Namespace):
    # 動画1本またはバッチ処理のメイン処理
    os.makedirs(args.output, exist_ok=True)
    means_csv = os.path.join(args.output, 'corner_means.csv')
    write_header = not os.path.exists(means_csv)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f'Failed to open video: {args.video}')

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Optional overlay video writer
    overlay_writer = None
    # Resolve overlay video path: allow passing a directory or filename without extension
    if args.overlay_video:
        if os.path.isdir(args.overlay_video):
            overlay_video_path = os.path.join(args.overlay_video, 'overlay.mp4')
        else:
            root, ext = os.path.splitext(args.overlay_video)
            overlay_video_path = args.overlay_video if ext else (args.overlay_video + '.mp4')
    else:
        overlay_video_path = os.path.join(args.output, 'overlay.mp4')

    if args.save_overlay_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        overlay_writer = cv2.VideoWriter(overlay_video_path, fourcc, fps, (width, height))
        if not overlay_writer.isOpened():
            raise RuntimeError(
                f"Failed to open VideoWriter for '{overlay_video_path}'. "
                f"Ensure the path is a valid filename (e.g., '.../overlay.mp4'), not a directory, and that the codec is supported."
            )

    buf = []
    idx = 0
    read_count = 0
    last_field = None  # keep latest interpolated field for per-frame overlay
    # Prepare optional per-frame saving
    frame_index = 0
    overlay_frames_dir = None
    raw_frames_dir = None
    if getattr(args, 'save_overlay_frames', False):
        overlay_frames_dir = os.path.join(args.output, 'frames_overlay')
        os.makedirs(overlay_frames_dir, exist_ok=True)
    if getattr(args, 'save_raw_frames', False):
        raw_frames_dir = os.path.join(args.output, 'frames_raw')
        os.makedirs(raw_frames_dir, exist_ok=True)

    with open(means_csv, 'a', newline='') as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=['frame', 'bl_x', 'bl_y', 'br_x', 'br_y', 'tl_x', 'tl_y', 'tr_x', 'tr_y'])
        if write_header:
            writer.writeheader()

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            read_count += 1
            buf.append(frame)

            # progress
            if not args.no_progress and (read_count % max(1, args.progress_interval) == 0):
                prefix = ""
                if hasattr(args, '_overall_total') and getattr(args, '_overall_total', 0) > 0:
                    overall_pct = 100.0 * (getattr(args, '_overall_done', 0) + read_count) / float(getattr(args, '_overall_total'))
                    prefix = f"[overall {overall_pct:5.1f}%] "
                if total_frames > 0:
                    pct = 100.0 * read_count / total_frames
                    print(f"{prefix}Frames: {read_count}/{total_frames} ({pct:.1f}%) | Pairs: {idx}", end='\r', flush=True)
                else:
                    print(f"{prefix}Frames: {read_count} | Pairs: {idx}", end='\r', flush=True)

            # Optional save raw frame
            if raw_frames_dir is not None:
                cv2.imwrite(os.path.join(raw_frames_dir, f"frame_{frame_index+1:06d}.png"), frame)

            if len(buf) < args.frame_gap + 1:
                # Even before first pair, we can still write raw frame (no arrows) to video if requested
                if overlay_writer is not None:
                    overlay_writer.write(frame)
                frame_index += 1
                continue

            # stride decision
            should_emit = (args.stride <= 1) or ((idx % max(1, args.stride)) == 0)
            if should_emit:
                I1 = buf[0]
                I2 = buf[-1]
                crops1 = get_corner_crops(I1, args.region_ratio)
                crops2 = get_corner_crops(I2, args.region_ratio)

                means: Dict[str, Tuple[float, float]] = {}
                # Decide backend (torch if requested and available)
                use_torch = False
                device = 'cpu'
                if args.device == 'cuda' and TORCH_AVAILABLE and torch.cuda.is_available():
                    use_torch = True
                    device = 'cuda'
                elif args.device == 'auto' and TORCH_AVAILABLE and torch.cuda.is_available():
                    use_torch = True
                    device = 'cuda'

                if use_torch:
                    I1_list = [crops1[r] for r in REGION_NAMES]
                    I2_list = [crops2[r] for r in REGION_NAMES]
                    flows = hs_pyramid_flow_torch_batch(I1_list, I2_list,
                                                        alpha=args.alpha, iters=args.iters,
                                                        outer_iters=args.outer_iters,
                                                        pyr_levels=args.pyr_levels,
                                                        pyr_scale=args.pyr_scale,
                                                        device=device, mixed_precision=args.mixed_precision)
                    for region, flow in zip(REGION_NAMES, flows):
                        fx = flow[..., 0]
                        fy = flow[..., 1]
                        mu_x, mu_y = trimmed_mean_uv_by_magnitude(fx, fy, args.trim_low_frac, args.trim_high_frac)
                        means[region] = (mu_x, mu_y)
                else:
                    for region in REGION_NAMES:
                        flow = hs_pyramid_flow(crops1[region], crops2[region],
                                               alpha=args.alpha, iters=args.iters,
                                               outer_iters=args.outer_iters,
                                               pyr_levels=args.pyr_levels,
                                               pyr_scale=args.pyr_scale)
                        fx = flow[..., 0]
                        fy = flow[..., 1]
                        mu_x, mu_y = trimmed_mean_uv_by_magnitude(fx, fy, args.trim_low_frac, args.trim_high_frac)
                        means[region] = (mu_x, mu_y)

                # Write CSV row (frame number approximated as idx*gap)
                writer.writerow({
                    'frame': idx * args.frame_gap,
                    'bl_x': means['bl'][0], 'bl_y': means['bl'][1],
                    'br_x': means['br'][0], 'br_y': means['br'][1],
                    'tl_x': means['tl'][0], 'tl_y': means['tl'][1],
                    'tr_x': means['tr'][0], 'tr_y': means['tr'][1],
                })

                if args.save_field or args.save_overlay or args.save_overlay_video:
                    field = bilinear_field(height, width, means['tl'], means['tr'], means['bl'], means['br'])
                    last_field = field
                    if args.save_field:
                        npy_path = os.path.join(args.output, f'field_{idx:04d}.npy')
                        np.save(npy_path, field.astype(np.float32))
                    if args.save_overlay:
                        overlay = draw_grid_arrows(I2, field, step=args.grid_step, scale=args.scale, min_pixels=args.min_pixels)
                        cv2.imwrite(os.path.join(args.output, f'overlay_{idx:04d}.png'), overlay)

            # Write overlay video frame (draw with latest field if available)
            if overlay_writer is not None:
                if last_field is not None:
                    frame_to_write = draw_grid_arrows(frame, last_field, step=args.grid_step, scale=args.scale, min_pixels=args.min_pixels)
                else:
                    frame_to_write = frame
                overlay_writer.write(frame_to_write)
                # Optional save overlay frame as image
                if overlay_frames_dir is not None:
                    cv2.imwrite(os.path.join(overlay_frames_dir, f"frame_{frame_index+1:06d}.png"), frame_to_write)

            # If not writing overlay video but user requested overlay frames, still save
            elif overlay_frames_dir is not None:
                if last_field is not None:
                    frame_to_write = draw_grid_arrows(frame, last_field, step=args.grid_step, scale=args.scale, min_pixels=args.min_pixels)
                else:
                    frame_to_write = frame
                cv2.imwrite(os.path.join(overlay_frames_dir, f"frame_{frame_index+1:06d}.png"), frame_to_write)

            idx += 1
            frame_index += 1

            if args.non_overlap:
                buf = [buf[-1]]
            else:
                buf.pop(0)

    cap.release()
    if overlay_writer is not None:
        overlay_writer.release()
    if not args.no_progress:
        print()  # newline
        print(f"Done. Frames read: {read_count}. Pairs emitted: {idx}." )
        if args.save_overlay_video:
            print(f"Overlay video saved to: {overlay_video_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='カメラモーション推定（コーナーHS＋バイリニア補間）')
    # 動画1本またはバッチフォルダ
    p.add_argument('--video', help='入力動画のパス')
    p.add_argument('--input_dir', help='このフォルダ内の全動画を処理（--outputと併用）。指定時は--videoは無視される')
    p.add_argument('--recursive', action='store_true', help='input_dirを再帰的にスキャン')
    p.add_argument('--exts', nargs='*', default=['.mp4', '.mov', '.avi', '.mkv', '.m4v', '.wmv', '.webm'], help='対象とする動画拡張子')

    p.add_argument('--output', default=os.path.join('Scripts', 'outputs_camera_motion_hs'), help='出力ルートディレクトリ')
    p.add_argument('--region_ratio', type=float, default=0.10, help='コーナーパッチの幅・高さ比率')
    p.add_argument('--frame_gap', type=int, default=10, help='ペア間のフレームギャップ（i→i+gap）')
    p.add_argument('--stride', type=int, default=1, help='Nペアごとに出力（スライディングウィンドウ）')
    p.add_argument('--non_overlap', action='store_true', help='重複なしウィンドウ（gap分進める）')

    # HSピラミッドパラメータ
    p.add_argument('--alpha', type=float, default=10.0)
    p.add_argument('--iters', type=int, default=50)
    p.add_argument('--outer_iters', type=int, default=3)
    p.add_argument('--pyr_levels', type=int, default=4)
    p.add_argument('--pyr_scale', type=float, default=0.5)

    # オーバーレイ・グリッドオプション
    p.add_argument('--save_field', action='store_true', help='ペアごとに密な補間フィールドを.npy保存')
    p.add_argument('--save_overlay', action='store_true', help='フィールドのグリッド矢印画像を保存')
    p.add_argument('--save_overlay_video', action='store_true', help='矢印オーバーレイ動画（overlay.mp4）を保存')
    p.add_argument('--overlay_video', help='オーバーレイ動画の出力パス（mp4）。デフォルト: <output>/overlay.mp4')
    p.add_argument('--grid_step', type=int, default=40, help='グリッド描画のサンプリング間隔')
    p.add_argument('--scale', type=float, default=20.0, help='矢印長さスケール')
    p.add_argument('--min_pixels', type=float, default=0.0, help='矢印の最小表示長（ピクセル）')
    p.add_argument('--save_overlay_frames', action='store_true', help='フレームごとのオーバーレイ画像もframes_overlay/に保存')
    p.add_argument('--save_raw_frames', action='store_true', help='フレームごとの生画像もframes_raw/に保存')

    # ロバスト平均化オプション（コーナーパッチごと）
    p.add_argument('--trim_low_frac', type=float, default=0.2, help='平均前に最小大きさ部分を除外する割合（例: 0.2）')
    p.add_argument('--trim_high_frac', type=float, default=0.2, help='平均前に最大大きさ部分を除外する割合（例: 0.2）')

    # デバイス・高速化
    p.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto', help='HSのバックエンド選択（autoはCUDAがあれば使用）')
    p.add_argument('--mixed_precision', action='store_true', help='CUDAでAMPを使い高速化（torchバックエンドのみ）')

    # 進捗表示
    p.add_argument('--no_progress', action='store_true')
    p.add_argument('--progress_interval', type=int, default=50)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    # バッチモード
    if args.input_dir:
        in_dir = os.path.abspath(args.input_dir)
        out_root = os.path.abspath(args.output)
        if not os.path.isdir(in_dir):
            raise RuntimeError(f'input_dirが見つかりません: {in_dir}')
        os.makedirs(out_root, exist_ok=True)

        vids = list_videos(in_dir, args.exts, args.recursive)
        if not vids:
            print(f'動画が見つかりません: {in_dir}')
            raise SystemExit(0)

        # 総フレーム数を事前取得
        total_known = 0
        for vp in vids:
            cap0 = cv2.VideoCapture(vp)
            if cap0.isOpened():
                n = int(cap0.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                if n > 0:
                    total_known += n
            cap0.release()

        processed_known = 0
        print(f'{len(vids)}本の動画を検出。総フレーム数: {total_known if total_known>0 else "不明"}')
        for i, vp in enumerate(vids, start=1):
            base = video_base_name(vp)
            out_dir = os.path.join(out_root, base)
            os.makedirs(out_dir, exist_ok=True)
            # argsのシャローコピー
            from argparse import Namespace
            local_args = Namespace(**vars(args))
            local_args.video = vp
            local_args.output = out_dir
            if total_known > 0:
                local_args._overall_total = total_known
                local_args._overall_done = processed_known
            print(f"\n[{i}/{len(vids)}] {vp} -> {out_dir}")
            process(local_args)
            # この動画のフレーム数分進捗を進める
            cap1 = cv2.VideoCapture(vp)
            if cap1.isOpened():
                processed_known += int(cap1.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            cap1.release()
        print('\n完了。')
    else:
        if not args.video:
            raise RuntimeError('--videoまたは--input_dirを指定してください')
        process(args)
