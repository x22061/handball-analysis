'''
コート画像に9mライン（仕様の四分円＋直線）を赤線で描画するスクリプト。
9mライン定義の確認用のプログラムです。

- 座標系は既存データと同じく正規化座標 [0,1]×[0,1] を前提とし、
  画像ピクセルへは x_px=W*x, y_px=H*y で写像する。
  半径 r=0.45 も x 方向は W*r, y 方向は H*r としてスケールする（非正方画像では楕円になる）。

描画仕様（与えられた問題文に準拠）:
    left:
        - 中心 (0, 0.425) 半径 0.45 の右上 1/4 円弧
        - 中心 (0, 0.575) 半径 0.45 の右下 1/4 円弧
    - それらの端点を結ぶ直線（x=0.45, y∈[0.425, 0.575]）
    right:
        - 中心 (1, 0.425) 半径 0.45 の左上 1/4 円弧
        - 中心 (1, 0.575) 半径 0.45 の左下 1/4 円弧
    - それらの端点を結ぶ直線（x=0.55, y∈[0.425, 0.575]）

使い方 (Windows PowerShell):
    - 1枚の画像に描く場合:
        python Scripts/draw_9m_line.py --image data/court.png --side both --output data/court_with_9m.png
    - left/right で別画像に個別描画する場合（両方指定可）:（1行で）
        python Scripts/draw_9m_line.py --left-image data/court_left.png --left-output data/court_left_9m.png --right-image data/court_right.png --right-output data/court_right_9m.png
      出力を省略した場合は自動命名（*_with_9m_left / *_with_9m_right）。
    - 画像がない場合は空の背景を作って描くことも可能:
        python Scripts/draw_9m_line.py --blank 1280x720 --side both --output data/blank_with_9m.png

必要ライブラリ: Pillow (PIL)
'''

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image, ImageDraw


Color = Tuple[int, int, int]


# 正規化→ピクセル座標への変換
def nx(x: float, W: int) -> float:
    return x * float(W)


def ny(y: float, H: int) -> float:
    return y * float(H)


def draw_left_9m(draw: ImageDraw.ImageDraw, W: int, H: int, color: Color = (255, 0, 0), width: int = 4) -> None:
    """left 側の9mラインを描画（右上/右下の四分円＋x=0.45の直線）。"""
    r = 0.45
    # 上側四分円（中心 (0, 0.425)）: 角度 270→360（右上）
    cx_up, cy_up = nx(0.0, W), ny(0.425, H)
    rx, ry = nx(r, W), ny(r, H)
    bbox_up = (cx_up - rx, cy_up - ry, cx_up + rx, cy_up + ry)
    draw.arc(bbox_up, start=270, end=360, fill=color, width=width)

    # 下側四分円（中心 (0, 0.575)）: 角度 0→90（右下）
    cx_lo, cy_lo = nx(0.0, W), ny(0.575, H)
    bbox_lo = (cx_lo - rx, cy_lo - ry, cx_lo + rx, cy_lo + ry)
    draw.arc(bbox_lo, start=0, end=90, fill=color, width=width)

    # 直線 x = 0.45, y ∈ [0.425, 0.575]
    x_line = nx(0.45, W)
    y1, y2 = ny(0.425, H), ny(0.575, H)
    draw.line([(x_line, y1), (x_line, y2)], fill=color, width=width)


def draw_right_9m(draw: ImageDraw.ImageDraw, W: int, H: int, color: Color = (255, 0, 0), width: int = 4) -> None:
    """right 側の9mラインを描画（左上/左下の四分円＋x=0.55の直線）。"""
    r = 0.45
    # 上側四分円（中心 (1, 0.425)）: 角度 180→270（左上）
    cx_up, cy_up = nx(1.0, W), ny(0.425, H)
    rx, ry = nx(r, W), ny(r, H)
    bbox_up = (cx_up - rx, cy_up - ry, cx_up + rx, cy_up + ry)
    draw.arc(bbox_up, start=180, end=270, fill=(255, 0, 0), width=width)

    # 下側四分円（中心 (1, 0.575)）: 角度 90→180（左下）
    cx_lo, cy_lo = nx(1.0, W), ny(0.575, H)
    bbox_lo = (cx_lo - rx, cy_lo - ry, cx_lo + rx, cy_lo + ry)
    draw.arc(bbox_lo, start=90, end=180, fill=(255, 0, 0), width=width)

    # 直線 x = 0.55, y ∈ [0.425, 0.575]
    x_line = nx(0.55, W)
    y1, y2 = ny(0.425, H), ny(0.575, H)
    draw.line([(x_line, y1), (x_line, y2)], fill=(255, 0, 0), width=width)


def parse_size(s: str) -> Tuple[int, int]:
    """'1280x720' のような表記を (1280,720) にパース。"""
    parts = s.lower().split('x')
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("サイズは WxH の形式で指定してください（例: 1280x720）")
    try:
        w = int(parts[0])
        h = int(parts[1])
    except ValueError:
        raise argparse.ArgumentTypeError("サイズは整数で指定してください（例: 1280x720）")
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError("サイズは正の整数で指定してください")
    return w, h

def compute_line_width(W: int, H: int) -> int:
    """画像サイズから線幅の目安を決める（最小2px）。"""
    base = max(2, int(0.004 * min(W, H)))
    return base


def _process_image(src_path: Path, side: str, out_path: Optional[Path] = None) -> Path:
    """単一画像に指定サイドの9mラインを描画し、保存してパスを返す。"""
    img = Image.open(src_path).convert('RGB')
    W, H = img.size
    draw = ImageDraw.Draw(img)
    width = compute_line_width(W, H)
    if side == 'left':
        draw_left_9m(draw, W, H, color=(255, 0, 0), width=width)
        suffix = '_with_9m_left'
    elif side == 'right':
        draw_right_9m(draw, W, H, color=(255, 0, 0), width=width)
        suffix = '_with_9m_right'
    else:  # both
        draw_left_9m(draw, W, H, color=(255, 0, 0), width=width)
        draw_right_9m(draw, W, H, color=(255, 0, 0), width=width)
        suffix = '_with_9m'

    if out_path is None:
        out_path = src_path.with_name(src_path.stem + suffix + src_path.suffix)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description='コート画像に9mラインを描画して保存します。')
    # 単一画像モード
    parser.add_argument('--image', type=str, help='入力画像パス（指定しない場合は --blank を利用）')
    parser.add_argument('--blank', type=parse_size, help='空の画像を生成して描画（形式: WxH 例: 1280x720）')
    parser.add_argument('--side', type=str, default='both', choices=['left', 'right', 'both'], help='描画対象サイド')
    parser.add_argument('--output', type=str, help='出力画像パス（省略時は自動命名）')
    # 複数画像モード（left/right 個別）
    parser.add_argument('--left-image', type=str, help='left サイド用の入力画像パス')
    parser.add_argument('--right-image', type=str, help='right サイド用の入力画像パス')
    parser.add_argument('--left-output', type=str, help='left サイド用の出力画像パス（省略時は *_with_9m_left）')
    parser.add_argument('--right-output', type=str, help='right サイド用の出力画像パス（省略時は *_with_9m_right）')

    args = parser.parse_args()

    # 複数画像モード（優先）
    left_img_path = Path(args.left_image) if args.left_image else None
    right_img_path = Path(args.right_image) if args.right_image else None

    if left_img_path or right_img_path:
        if left_img_path:
            if not left_img_path.exists():
                raise FileNotFoundError(f'left 用入力画像が見つかりません: {left_img_path}')
            left_out = Path(args.left_output) if args.left_output else None
            outp = _process_image(left_img_path, side='left', out_path=left_out)
            print(f'left 9mラインを描画して保存しました: {outp}')
        if right_img_path:
            if not right_img_path.exists():
                raise FileNotFoundError(f'right 用入力画像が見つかりません: {right_img_path}')
            right_out = Path(args.right_output) if args.right_output else None
            outp = _process_image(right_img_path, side='right', out_path=right_out)
            print(f'right 9mラインを描画して保存しました: {outp}')
        return

    # 単一画像モード/ブランク生成
    img: Optional[Image.Image] = None
    src_path: Optional[Path] = Path(args.image) if args.image else None

    if src_path is not None:
        if not src_path.exists():
            raise FileNotFoundError(f'入力画像が見つかりません: {src_path}')
        img = Image.open(src_path).convert('RGB')
    elif args.blank is not None:
        w, h = args.blank
        # コート風の背景色（薄い緑）
        img = Image.new('RGB', (w, h), color=(210, 230, 210))
    else:
        raise SystemExit('画像を指定してください: --image <path> もしくは --blank WxH あるいは --left-image/--right-image')

    W, H = img.size
    draw = ImageDraw.Draw(img)
    width = compute_line_width(W, H)

    if args.side in ('left', 'both'):
        draw_left_9m(draw, W, H, color=(255, 0, 0), width=width)
    if args.side in ('right', 'both'):
        draw_right_9m(draw, W, H, color=(255, 0, 0), width=width)

    # 出力先決定
    if args.output:
        out_path = Path(args.output)
    else:
        if src_path is not None:
            suffix = '_with_9m' if args.side == 'both' else f"_with_9m_{args.side}"
            out_path = src_path.with_name(src_path.stem + suffix + src_path.suffix)
        else:
            out_path = Path('data') / 'blank_with_9m.png'
            out_path.parent.mkdir(parents=True, exist_ok=True)

    img.save(out_path)
    print(f'9mラインを描画して保存しました: {out_path}')


if __name__ == '__main__':
    main()
