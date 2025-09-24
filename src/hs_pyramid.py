"""
Horn–Schunck法でオプティカルフロー（ピラミッド型HS、反復ワーピング付き）の計算。

- 両フレームのガウシアンピラミッドを構築
- 最も粗いレベルからゼロフローで開始し、解像度を上げながら精緻化
- 各レベルでワーピングとHorn–Schunck更新を交互に実行

ペアごとの出力:
- flow_color_XXXX.png  : HSVカラー可視化画像
- flow_XXXX.npy        : 生のフロー（H x W x 2, float32）
- flow_means_full.csvに平均値: frame, mean_x, mean_y, std_x, std_y

使用例 (PowerShell):
python Scripts/hs_pyramid.py `
  --video "D:\data\video\48dcd3_00-06-00.mp4" `
  --output "D:\data\outputs_hs_pyramid" `
  --frame_gap 10 `
  --alpha 10.0 `
  --iters 50 `
  --outer_iters 3 `
  --pyr_levels 4 `
  --pyr_scale 0.5
"""

import os
import csv
import math
import argparse
from typing import Tuple

import cv2
import numpy as np


def gaussian_pyramid(img: np.ndarray, levels: int, scale: float) -> list[np.ndarray]:
    # 画像のガウシアンピラミッドを生成する関数
    pyr = [img]
    for _ in range(1, max(1, levels)):
        img = cv2.pyrDown(img) if abs(scale - 0.5) < 1e-6 else cv2.resize(
            img, (max(1, int(img.shape[1] * scale)), max(1, int(img.shape[0] * scale))), interpolation=cv2.INTER_AREA
        )
        pyr.append(img)
    return pyr


def warp_image(I: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    # フロー(u, v)に従って画像Iをワープする関数
    h, w = I.shape[:2]
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = (xx + u).astype(np.float32)
    map_y = (yy + v).astype(np.float32)
    return cv2.remap(I, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)


def horn_schunck_step(I1: np.ndarray, I2w: np.ndarray, u: np.ndarray, v: np.ndarray, alpha: float, iters: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    1レベルのHorn–Schunck反復（ワープ済み画像I2wを使用）。
    I1, I2wはグレースケールfloat32で[0,1]。
    """
    # 微分（I1とI2wの勾配の平均を使い、頑健性を向上）
    Ix1 = cv2.Sobel(I1, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    Iy1 = cv2.Sobel(I1, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    Ix2 = cv2.Sobel(I2w, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    Iy2 = cv2.Sobel(I2w, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    Ix = 0.5 * (Ix1 + Ix2)
    Iy = 0.5 * (Iy1 + Iy2)
    It = I2w - I1

    # 局所平均化カーネル（Horn–Schunckはラプラシアン風の平滑化を使用）
    kernel = np.array([[1, 2, 1],
                       [2, 4, 2],
                       [1, 2, 1]], dtype=np.float32) / 16.0

    alpha2 = alpha * alpha
    for _ in range(max(1, iters)):
        u_bar = cv2.filter2D(u, -1, kernel, borderType=cv2.BORDER_DEFAULT)
        v_bar = cv2.filter2D(v, -1, kernel, borderType=cv2.BORDER_DEFAULT)

        # 共通項
        der = Ix * u_bar + Iy * v_bar + It
        denom = alpha2 + Ix * Ix + Iy * Iy
        # 0除算を回避
        denom = np.where(denom <= 1e-6, 1e-6, denom)

        u = u_bar - (Ix * der) / denom
        v = v_bar - (Iy * der) / denom

    return u, v


def hs_pyramid_flow(I1: np.ndarray, I2: np.ndarray, alpha: float = 10.0, iters: int = 50,
                     outer_iters: int = 3, pyr_levels: int = 4, pyr_scale: float = 0.5) -> np.ndarray:
    """
    I1→I2のフローを粗から細へのHS法＋反復ワーピングで計算。
    戻り値はフロー（H x W x 2, float32）。
    """
    # float32グレースケール[0,1]に変換
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

    # 最も粗いレベルから開始
    L = len(pyr1)
    u = np.zeros_like(pyr1[-1], dtype=np.float32)
    v = np.zeros_like(pyr1[-1], dtype=np.float32)

    for level in range(L - 1, -1, -1):
        I1l = pyr1[level]
        I2l = pyr2[level]

        # フローを現在のレベルにアップスケール（最粗以外）
        if u.shape != I1l.shape:
            u = cv2.resize(u, (I1l.shape[1], I1l.shape[0]), interpolation=cv2.INTER_LINEAR) * (1.0 / pyr_scale)
            v = cv2.resize(v, (I1l.shape[1], I1l.shape[0]), interpolation=cv2.INTER_LINEAR) * (1.0 / pyr_scale)

        # 外側のワーピング反復
        for _ in range(max(1, outer_iters)):
            I2w = warp_image(I2l, u, v)
            u, v = horn_schunck_step(I1l, I2w, u, v, alpha=alpha, iters=iters)

    flow = np.dstack([u, v]).astype(np.float32)
    # NaNや無限大を0に置換
    flow = np.nan_to_num(flow, nan=0.0, posinf=0.0, neginf=0.0)
    return flow


def flow_to_color(flow: np.ndarray) -> np.ndarray:
    # フローをHSVカラー画像に変換する関数
    u = flow[..., 0]
    v = flow[..., 1]
    mag, ang = cv2.cartToPolar(u, v, angleInDegrees=True)

    # 大きさをロバストに正規化
    mag_finite = mag[np.isfinite(mag)]
    if mag_finite.size > 0:
        hi = np.percentile(mag_finite, 99.0)
        if hi <= 0:
            hi = 1.0
        mag_n = np.clip(mag, 0, hi) / hi
    else:
        mag_n = np.zeros_like(mag)

    hsv = np.zeros((*mag.shape, 3), dtype=np.uint8)
    hsv[..., 0] = (ang / 2).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = (mag_n * 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def process_video(args: argparse.Namespace):
    # 動画をフレームごとに処理し、フロー画像や統計情報を出力する関数
    os.makedirs(args.output, exist_ok=True)
    means_csv = os.path.join(args.output, 'flow_means_full.csv')
    write_header = not os.path.exists(means_csv)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f'Failed to open video: {args.video}')

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    idx = 0
    read_count = 0
    frame_buf: list[np.ndarray] = []

    with open(means_csv, 'a', newline='') as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=['frame', 'mean_x', 'mean_y', 'std_x', 'std_y'])
        if write_header:
            writer.writeheader()

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            read_count += 1
            frame_buf.append(frame)

            if len(frame_buf) < args.frame_gap + 1:
                # 進捗表示
                if not args.no_progress and (read_count % max(1, args.progress_interval) == 0):
                    if total_frames > 0:
                        pct = 100.0 * read_count / total_frames
                        print(f"Frames: {read_count}/{total_frames} ({pct:.1f}%) | Pairs: {idx}", end='\r', flush=True)
                    else:
                        print(f"Frames: {read_count} | Pairs: {idx}", end='\r', flush=True)
                continue

            # strideに基づき計算するか判定
            pair_ready = False
            if args.stride <= 1:
                pair_ready = True
            else:
                if ((idx % args.stride) == 0):
                    pair_ready = True

            if pair_ready:
                I1 = frame_buf[0]
                I2 = frame_buf[-1]
                flow = hs_pyramid_flow(I1, I2, alpha=args.alpha, iters=args.iters,
                                       outer_iters=args.outer_iters, pyr_levels=args.pyr_levels,
                                       pyr_scale=args.pyr_scale)
                color = flow_to_color(flow)
                cv2.imwrite(os.path.join(args.output, f'flow_color_{idx:04d}.png'), color)
                np.save(os.path.join(args.output, f'flow_{idx:04d}.npy'), flow)

                # フローの平均・標準偏差をCSVに記録
                fx = flow[..., 0].reshape(-1)
                fy = flow[..., 1].reshape(-1)
                writer.writerow({
                    'frame': idx * args.frame_gap,
                    'mean_x': float(np.mean(fx)),
                    'mean_y': float(np.mean(fy)),
                    'std_x': float(np.std(fx)),
                    'std_y': float(np.std(fy)),
                })

            idx += 1

            if args.non_overlap:
                # ウィンドウをframe_gap分進める
                frame_buf = [frame_buf[-1]]
            else:
                # ウィンドウを1フレーム分スライド
                frame_buf.pop(0)

            if not args.no_progress and (read_count % max(1, args.progress_interval) == 0):
                if total_frames > 0:
                    # ペア数の概算
                    approx_pairs = max(0, total_frames - args.frame_gap)
                    s = max(1, args.stride)
                    approx_pairs = (approx_pairs + s - 1) // s
                    pct_pairs = (100.0 * min(idx, approx_pairs) / approx_pairs) if approx_pairs > 0 else 0.0
                    print(f"Frames: {read_count}/{total_frames} ({(100.0*read_count/total_frames):.1f}%) | Pairs: {idx}/{approx_pairs} ({pct_pairs:.1f}%)", end='\r', flush=True)
                else:
                    print(f"Frames: {read_count} | Pairs: {idx}", end='\r', flush=True)

    cap.release()
    if not args.no_progress:
        print()  # 改行
        print(f"完了。読み込んだフレーム数: {read_count}。出力したペア数: {idx}。")


def parse_args() -> argparse.Namespace:
    # コマンドライン引数を解析する関数
    p = argparse.ArgumentParser(description='ピラミッド型Horn–Schunckオプティカルフロー（ワーピング付き）')
    p.add_argument('--video', required=True, help='入力動画のパス')
    p.add_argument('--output', default=os.path.join('Scripts', 'outputs_hs_pyramid'), help='出力ディレクトリ')
    p.add_argument('--frame_gap', type=int, default=10, help='ペア間のフレームギャップ（i→i+gapのフロー）')
    p.add_argument('--stride', type=int, default=1, help='Nペアごとに出力（スライディングウィンドウ）')
    p.add_argument('--non_overlap', action='store_true', help='ウィンドウを重複なしで進める（frame_gap分進める）')

    # HSパラメータ
    p.add_argument('--alpha', type=float, default=10.0, help='Horn–Schunckの平滑化重み（大きいほど滑らか）')
    p.add_argument('--iters', type=int, default=50, help='ワーピングごとのHS反復回数')
    p.add_argument('--outer_iters', type=int, default=3, help='ピラミッドレベルごとの外側ワーピング反復回数')

    # ピラミッドパラメータ
    p.add_argument('--pyr_levels', type=int, default=4, help='ピラミッドレベル数（粗から細へ）')
    p.add_argument('--pyr_scale', type=float, default=0.5, help='レベル間の縮小率（0.5ならpyrDown）')

    # 進捗表示
    p.add_argument('--no_progress', action='store_true', help='進捗表示を無効化')
    p.add_argument('--progress_interval', type=int, default=50, help='Nフレームごとに進捗を更新')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    process_video(args)
