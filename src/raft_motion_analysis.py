"""
RAFTを使ってflow, camera, localを抽出するスクリプトです。

【主な処理内容】
- 1つまたは複数の動画の隣接フレームペアに対してRAFTでオプティカルフローを計算
- 画像の四隅（各10%領域）のフローからカメラ運動を推定（上位/下位20%の大きさを除外したトリム平均）
- カメラ運動を補間（四隅→辺→全体）。--camera_interpで直接バイリニア補間も選択可能
- 必要に応じて各ペアごとの出力（生フロー/カメラ/ローカルフローのnpy/png、オーバレイ動画）を保存

【新機能】
- --input_dirでフォルダ内のmp4を一括処理（再帰/非再帰選択可）
- --output_rootで動画ごとにサブフォルダ作成（デフォルトはこのスクリプトのoutputs）
- 出力種別ごとに保存ON/OFF切り替え可能
- 既存出力をスキップして途中再開可能
- tqdmによる進捗表示（なければテキスト表示）

【使用例】
python raft_motion_analysis.py `
--model path/to/raft-sintel.pth `
--input_dir path/to/mp4s `
--output_root path/to/outputs `
--save_flow_png `
--save_camera_png `
--save_local_png `
--save_overlay `
--skip_existing

"""

import sys
import os
# 'core'ディレクトリをパスに追加（ローカルimportのため）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CORE_DIR = os.path.join(SCRIPT_DIR, 'core')
if CORE_DIR not in sys.path:
    sys.path.insert(0, CORE_DIR)

import argparse
import cv2
import numpy as np
import torch
import glob
import threading
import queue
from typing import List, Tuple

try:
    from tqdm import tqdm
except Exception:  # tqdmが無い場合は進捗バー無し
    tqdm = None

from raft import RAFT
from utils.utils import InputPadder
from utils import flow_viz

# デフォルトの出力先（このスクリプトのoutputsフォルダ下）
OUTPUT_ROOT_DEFAULT = os.path.join(SCRIPT_DIR, 'outputs')
os.makedirs(OUTPUT_ROOT_DEFAULT, exist_ok=True)

def get_device():
    # CUDAが使える場合はGPU、それ以外はCPU
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def setup_gpu_optimizations():
    """GPU最適化の設定（パフォーマンス向上用）"""
    if torch.cuda.is_available():
        # 入力サイズが一定ならcudnnのベンチマークを有効化
        torch.backends.cudnn.benchmark = True
        # メモリ断片化防止
        torch.cuda.empty_cache()

def frame_to_tensor(frame_bgr: np.ndarray, device: str, use_half: bool = False) -> torch.Tensor:
    # BGR画像をRGBに変換し、テンソル化（float32/float16, 1xCxHxW）
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    # CPU→GPU転送を高速化するためpinned memoryを使用
    tens = torch.from_numpy(rgb).pin_memory().permute(2, 0, 1).float()[None]
    if use_half:
        tens = tens.half()
    return tens.to(device, non_blocking=True)

def compute_raft_flow(model, img1: torch.Tensor, img2: torch.Tensor, iters: int = 20, 
                     padder: InputPadder = None) -> np.ndarray:
    # padderを使い回して高速化
    if padder is None:
        padder = InputPadder(img1.shape)
    i1, i2 = padder.pad(img1, img2)
    
    # 推論時はinference_modeで高速化
    with torch.inference_mode():
        _, flow_up = model(i1, i2, iters=iters, test_mode=True)
    
    # パディングを外してHxWx2のnumpy配列に変換
    flow = padder.unpad(flow_up[0]).permute(1, 2, 0).contiguous().cpu().numpy()
    return flow

def corner_roi_sizes(h: int, w: int):
    # 四隅ROIは画像面積の10%ずつ→一辺は√0.10倍
    import math
    scale = math.sqrt(0.10)
    h_roi = max(1, int(round(h * scale)))
    w_roi = max(1, int(round(w * scale)))
    return h_roi, w_roi

def trimmed_mean_vec(flow_roi: np.ndarray) -> np.ndarray:
    """flow_roi: (h, w, 2)。ベクトルの大きさで上下20%を除外し、中央60%の平均を計算"""
    vecs = flow_roi.reshape(-1, 2)
    if vecs.shape[0] == 0:
        return np.zeros(2, dtype=np.float32)
    mags = np.linalg.norm(vecs, axis=1)
    if vecs.shape[0] < 10:
        # 点が少ない場合は単純平均
        mean_vec = vecs.mean(axis=0)
        return mean_vec.astype(np.float32)
    lo = np.percentile(mags, 20)
    hi = np.percentile(mags, 80)
    keep = (mags >= lo) & (mags <= hi)
    if not np.any(keep):
        mean_vec = vecs.mean(axis=0)
    else:
        mean_vec = vecs[keep].mean(axis=0)
    return mean_vec.astype(np.float32)

def bilinear_camera_field(h: int, w: int,
                         v_tl: np.ndarray, v_tr: np.ndarray,
                         v_bl: np.ndarray, v_br: np.ndarray) -> np.ndarray:
    """四隅のベクトルからバイリニア補間でHxWx2のカメラ運動場を生成"""
    ys = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    xs = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
    top = (1 - xs)[..., None] * v_tl[None, None, :] + xs[..., None] * v_tr[None, None, :]
    bot = (1 - xs)[..., None] * v_bl[None, None, :] + xs[..., None] * v_br[None, None, :]
    field = (1 - ys)[..., None] * top + ys[..., None] * bot
    return field.astype(np.float32)

def stagewise_camera_field(h: int, w: int,
                          v_tl: np.ndarray, v_tr: np.ndarray,
                          v_bl: np.ndarray, v_br: np.ndarray) -> np.ndarray:
    """論文方式：四隅→辺→全体の段階的補間でHxWx2のカメラ運動場を生成

    手順:
        1) 四隅から各辺（上・下はx方向、左・右はy方向）を補間
        2) 各行yごとに左→右のベクトルをx方向に補間して内部を埋める
    線形仮定下ではバイリニアと同値だが、段階的構成を明示
    """
    # エッジ補間
    xs = np.linspace(0.0, 1.0, w, dtype=np.float32)
    ys = np.linspace(0.0, 1.0, h, dtype=np.float32)

    # 上下エッジ（xの関数として）
    top = ((1 - xs)[:, None] * v_tl[None, :] + xs[:, None] * v_tr[None, :]).astype(np.float32)   # (w,2)
    bot = ((1 - xs)[:, None] * v_bl[None, :] + xs[:, None] * v_br[None, :]).astype(np.float32)   # (w,2)
    # 左右エッジ（yの関数として）
    left = ((1 - ys)[:, None] * v_tl[None, :] + ys[:, None] * v_bl[None, :]).astype(np.float32)  # (h,2)
    right = ((1 - ys)[:, None] * v_tr[None, :] + ys[:, None] * v_br[None, :]).astype(np.float32) # (h,2)

    # 内部補間: 各行で左[y]と右[y]をx方向に補間
    alpha = xs[None, :, None]  # (1,w,1)
    field = (1 - alpha) * left[:, None, :] + alpha * right[:, None, :]

    # 明示的に四隅の整合性を強制（形状を揃える）
    field[0, :, :] = top      # top: (w,2)
    field[-1, :, :] = bot     # bot: (w,2)
    field[:, 0, :] = left     # left: (h,2)
    field[:, -1, :] = right   # right: (h,2)
    return field.astype(np.float32)

class FrameReader:
    """バックグラウンドでフレームを読み込むクラス（I/O並列化用）"""
    
    def __init__(self, cap, max_queue_size=3):
        self.cap = cap
        self.frame_queue = queue.Queue(maxsize=max_queue_size)
        self.stop_event = threading.Event()
        self.thread = None
        
    def start(self):
        """バックグラウンド読み込みスレッドを開始"""
        self.thread = threading.Thread(target=self._read_frames)
        self.thread.daemon = True
        self.thread.start()
        
    def _read_frames(self):
        """バックグラウンドでフレームを読み込むスレッド"""
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                # 動画の終端を示す
                self.frame_queue.put((False, None))
                break
            try:
                # フレームをキューに追加（ブロックしないようにタイムアウト付き）
                self.frame_queue.put((True, frame), timeout=1.0)
            except queue.Full:
                # キューが満杯の場合はこのフレームをスキップ
                continue
                
    def get_frame(self):
        """キューから次のフレームを取得"""
        try:
            return self.frame_queue.get(timeout=5.0)  # 5秒タイムアウト
        except queue.Empty:
            return False, None
            
    def stop(self):
        """バックグラウンド読み込みを停止"""
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)

def draw_arrows(frame_bgr: np.ndarray, field: np.ndarray, grid_step: int = 40, scale: float = 10.0, thickness: int = 2) -> np.ndarray:
    # 画像上にベクトル場（矢印）を描画する関数
    h, w = frame_bgr.shape[:2]
    out = frame_bgr.copy()
    color = (0, 255, 0)
    tip = 0.3

    # 半ステップから開始して境界を滑らかに
    y0 = grid_step // 2
    x0 = grid_step // 2
    for y in range(y0, h, grid_step):
        for x in range(x0, w, grid_step):
            vx, vy = field[y, x]
            end_x = int(round(x + scale * vx))
            end_y = int(round(y + scale * vy))
            cv2.arrowedLine(out, (x, y), (end_x, end_y), color, thickness, tipLength=tip)
    return out

def list_videos(input_dir: str, recursive: bool) -> List[str]:
    # 指定フォルダ内のmp4動画一覧を取得（再帰/非再帰）
    pattern = os.path.join(input_dir, '**', '*.mp4') if recursive else os.path.join(input_dir, '*.mp4')
    vids = glob.glob(pattern, recursive=recursive)
    vids.sort()
    return vids

def ensure_dir(p: str):
    # ディレクトリがなければ作成
    os.makedirs(p, exist_ok=True)

def check_missing_outputs(out_dir: str, idx: int, args) -> dict:
    """各出力（npy/png）が未作成かどうかを判定して辞書で返す

    戻り値のキー:
        flow_npy, flow_png,
        camera_npy, camera_png,
        local_npy, local_png,
        need_any  （上記のいずれかが必要か）
        need_overlay （--save_overlayの指定に従う。camera_field再計算時のみ生成）

    判定方針:
        - フラグで要求されている拡張子だけ存在確認
        - flow / camera / localごとにnpyとpngを独立判定
        - 新ディレクトリ構造とレガシーファイル名のどちらかが存在すれば既存とみなす
    """
    result = {
        'flow_npy': False,
        'flow_png': False,
        'camera_npy': False,
        'camera_png': False,
        'local_npy': False,
        'local_png': False,
        'need_any': False,
        'need_overlay': False,
    }

    def exists_either(new_path: str, legacy_path: str) -> bool:
        return os.path.exists(new_path) or os.path.exists(legacy_path)

    # FLOW
    if args.save_flow_npy:
        if not exists_either(os.path.join(out_dir, 'flow', f'flow_{idx:04d}.npy'),
                             os.path.join(out_dir, f'flow_{idx:04d}.npy')):
            result['flow_npy'] = True
    if args.save_flow_png:
        if not exists_either(os.path.join(out_dir, 'flow', f'flow_{idx:04d}.png'),
                             os.path.join(out_dir, f'flow_{idx:04d}.png')):
            result['flow_png'] = True

    # CAMERA
    if args.save_camera_npy:
        if not exists_either(os.path.join(out_dir, 'camera', f'camera_field_{idx:04d}.npy'),
                             os.path.join(out_dir, f'camera_field_{idx:04d}.npy')):
            result['camera_npy'] = True
    if args.save_camera_png:
        if not exists_either(os.path.join(out_dir, 'camera', f'camera_field_{idx:04d}.png'),
                             os.path.join(out_dir, f'camera_field_{idx:04d}.png')):
            result['camera_png'] = True

    # LOCAL
    if args.save_local_npy:
        if not exists_either(os.path.join(out_dir, 'local', f'local_{idx:04d}.npy'),
                             os.path.join(out_dir, f'local_{idx:04d}.npy')):
            result['local_npy'] = True
    if args.save_local_png:
        if not exists_either(os.path.join(out_dir, 'local', f'local_{idx:04d}.png'),
                             os.path.join(out_dir, f'local_{idx:04d}.png')):
            result['local_png'] = True

    # Any needed?
    result['need_any'] = any([
        result['flow_npy'], result['flow_png'],
        result['camera_npy'], result['camera_png'],
        result['local_npy'], result['local_png']
    ])
    result['need_overlay'] = args.save_overlay
    return result

def process_one_video(args, model, device, video_path: str, output_root: str) -> Tuple[bool, str]:
    # 1本の動画を処理し、各種出力を保存する関数
    base = os.path.splitext(os.path.basename(video_path))[0]
    base = base.replace(' ', '_')
    out_dir = os.path.join(output_root, base)
    ensure_dir(out_dir)

    # カテゴリごとのサブフォルダ作成
    flow_dir = os.path.join(out_dir, 'flow')
    camera_dir = os.path.join(out_dir, 'camera')
    local_dir = os.path.join(out_dir, 'local')
    # 一貫性のため全て作成
    ensure_dir(flow_dir)
    ensure_dir(camera_dir)
    ensure_dir(local_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False, f"[ERROR] 動画を開けません: {video_path}"

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # オーバレイ用のライター（任意）
    writer = None
    overlay_path = os.path.join(out_dir, 'overlay_camera.mp4')
    if args.save_overlay:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(overlay_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            writer = None
            print(f"[WARN] オーバレイ動画の作成に失敗: {overlay_path}")

    # I/O並列化のためフレームリーダーをセットアップ
    frame_reader = FrameReader(cap, max_queue_size=3)
    frame_reader.start()
    
    # 最初のフレームを読み込み
    ok, prev_bgr = frame_reader.get_frame()
    if not ok:
        frame_reader.stop()
        if writer is not None:
            writer.release()
        cap.release()
        return False, f"[ERROR] 最初のフレームが読み込めません: {video_path}"

    # フレーム処理ループ
    idx = 0
    total_pairs = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) - 1
    iterator = range(10**9)  # ダミー; 読み込み失敗でブレーク
    if tqdm is not None:
        iterator = tqdm(iterator, total=max(total_pairs, 0), desc=f"{base}", unit='pair')

    try:
        # 最適化のためパッディング器を1回だけ作成
        padder = None
        use_half = args.mixed_precision
        skip_count = 0
        
        for _ in iterator:
            ok, curr_bgr = frame_reader.get_frame()
            if not ok:
                break

            # 出力の存在チェック（部分追加サポート）
            if args.skip_existing:
                missing = check_missing_outputs(out_dir, idx, args)
                if not missing['need_any'] and not missing['need_overlay']:
                    skip_count += 1
                    if args.skip_log_interval > 0 and (skip_count % args.skip_log_interval == 0):
                        print(f"[SKIP] {base}: スキップしたペア数 {skip_count}（最後のidx {idx:04d}）")
                    idx += 1
                    prev_bgr = curr_bgr
                    continue
                elif args.log_missing and missing['need_any']:
                    # このペアは処理されるので、具体的に何が欠けているか表示
                    missing_items = [k for k in ['flow_npy','flow_png','camera_npy','camera_png','local_npy','local_png'] if missing.get(k)]
                    if missing_items:
                        print(f"[MISSING] {base} idx={idx:04d} -> {', '.join(missing_items)}")
                        if args.debug_missing_paths:
                            def reconstruct_paths(kind: str):
                                if kind.startswith('flow'):
                                    stem = f"flow_{idx:04d}"
                                    subdir = 'flow'
                                elif kind.startswith('camera'):
                                    stem = f"camera_field_{idx:04d}"
                                    subdir = 'camera'
                                else:
                                    stem = f"local_{idx:04d}"
                                    subdir = 'local'
                                ext = '.npy' if kind.endswith('npy') else '.png'
                                new_path = os.path.join(out_dir, subdir, stem + ext)
                                legacy_path = os.path.join(out_dir, stem + ext)
                                return new_path, legacy_path
                            for kind in missing_items:
                                new_p, legacy_p = reconstruct_paths(kind)
                                print(f"    {kind}: new_exists={os.path.exists(new_p)} -> {new_p}")
                                print(f"           legacy_exists={os.path.exists(legacy_p)} -> {legacy_p}")
            else:
                # 全ての出力を未作成扱い（強制再生成）
                missing = {
                    'flow_npy': args.save_flow_npy,
                    'flow_png': args.save_flow_png,
                    'camera_npy': args.save_camera_npy,
                    'camera_png': args.save_camera_png,
                    'local_npy': args.save_local_npy,
                    'local_png': args.save_local_png,
                    'need_any': (args.save_flow_npy or args.save_flow_png or
                                 args.save_camera_npy or args.save_camera_png or
                                 args.save_local_npy or args.save_local_png),
                    'need_overlay': args.save_overlay
                }

            # RAFTフロー計算は必要な出力がある場合のみ実行
            flow = None
            camera_field = None
            local = None
            
            if missing['need_any']:
                # テンソル準備＆RAFTフロー計算（最適化あり）
                img1 = frame_to_tensor(prev_bgr, device, use_half)
                img2 = frame_to_tensor(curr_bgr, device, use_half)
                
                # 初回使用時にパッディング器を作成し、その後は使い回し
                if padder is None:
                    padder = InputPadder(img1.shape)
                
                flow = compute_raft_flow(model, img1, img2, iters=args.iters, padder=padder)
                h, w, _ = flow.shape

                # 四隅ROIとトリム平均
                h_roi, w_roi = corner_roi_sizes(h, w)
                tl = flow[0:h_roi, 0:w_roi]
                tr = flow[0:h_roi, w - w_roi:w]
                bl = flow[h - h_roi:h, 0:w_roi]
                br = flow[h - h_roi:h, w - w_roi:w]

                v_tl = trimmed_mean_vec(tl)
                v_tr = trimmed_mean_vec(tr)
                v_bl = trimmed_mean_vec(bl)
                v_br = trimmed_mean_vec(br)

                if args.camera_interp == 'bilinear':
                    camera_field = bilinear_camera_field(h, w, v_tl, v_tr, v_bl, v_br)
                else:
                    camera_field = stagewise_camera_field(h, w, v_tl, v_tr, v_bl, v_br)
                local = flow - camera_field

                # ローカル閾値処理: 小さい残差をゼロに
                if args.local_thresh > 0:
                    mags = np.linalg.norm(local, axis=2)
                    local[mags < args.local_thresh] = 0.0

            # 欠損している出力のみ保存
            # FLOWの保存
            if missing.get('flow_npy'):
                np.save(os.path.join(flow_dir, f'flow_{idx:04d}.npy'), flow)
            if missing.get('flow_png'):
                flow_img = flow_viz.flow_to_image(flow).astype(np.uint8)
                cv2.imwrite(os.path.join(flow_dir, f'flow_{idx:04d}.png'), cv2.cvtColor(flow_img, cv2.COLOR_RGB2BGR))

            # CAMERAの保存
            if missing.get('camera_npy'):
                np.save(os.path.join(camera_dir, f'camera_field_{idx:04d}.npy'), camera_field)
            if missing.get('camera_png'):
                cam_img = flow_viz.flow_to_image(camera_field).astype(np.uint8)
                cv2.imwrite(os.path.join(camera_dir, f'camera_field_{idx:04d}.png'), cv2.cvtColor(cam_img, cv2.COLOR_RGB2BGR))

            # LOCALの保存
            if missing.get('local_npy'):
                np.save(os.path.join(local_dir, f'local_{idx:04d}.npy'), local)
            if missing.get('local_png'):
                local_img = flow_viz.flow_to_image(local).astype(np.uint8)
                cv2.imwrite(os.path.join(local_dir, f'local_{idx:04d}.png'), cv2.cvtColor(local_img, cv2.COLOR_RGB2BGR))

            # オーバレイ（常に書き込み有効なら、計算または再計算されたcamera_fieldを使用）
            if writer is not None and camera_field is not None:
                overlay = draw_arrows(curr_bgr, camera_field, grid_step=args.grid_step, scale=args.scale, thickness=args.thickness)
                writer.write(overlay)

            prev_bgr = curr_bgr
            idx += 1
    finally:
        frame_reader.stop()
        if writer is not None:
            writer.release()
        cap.release()
        if args.skip_existing and 'skip_count' in locals():
            print(f"[INFO] {base}: スキップしたペアの合計 = {skip_count}")

    return True, f"[OK] {base}: ペア数={idx}, 出力先={out_dir}"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RAFTによるカメラ運動一括抽出')
    parser.add_argument('--model', required=True, help='RAFTの学習済みモデルのパス')
    # 入力指定: mp4フォルダ（推奨）または単一動画
    parser.add_argument('--input_dir', help='処理対象の.mp4ファイルが入ったフォルダ')
    parser.add_argument('--video', help='単一動画ファイル（後方互換）')
    parser.add_argument('--recursive', action='store_true', help='input_dirを再帰的に検索（サブフォルダ含む）')
    # 出力ルート（動画ごとにサブフォルダ作成）
    parser.add_argument('--output_root', default=OUTPUT_ROOT_DEFAULT, help='出力ルートディレクトリ')
    # RAFTオプション
    parser.add_argument('--small', action='store_true', help='小型モデルを使用')
    parser.add_argument('--mixed_precision', action='store_true', help='mixed precisionで高速化')
    parser.add_argument('--alternate_corr', action='store_true', help='効率的な相関計算を使用')
    parser.add_argument('--iters', type=int, default=20, help='RAFTの反復回数')
    # オーバレイ描画オプション
    parser.add_argument('--grid_step', type=int, default=40, help='矢印のグリッド間隔（ピクセル）')
    parser.add_argument('--scale', type=float, default=10.0, help='矢印のスケール')
    parser.add_argument('--thickness', type=int, default=2, help='矢印の線の太さ')
    # 出力種別ごとの保存ON/OFF
    parser.add_argument('--save_overlay', action='store_true', help='カメラベクトル付きオーバレイ動画を保存')
    parser.add_argument('--save_flow_npy', action='store_true', help='flow_XXXX.npyを保存')
    parser.add_argument('--save_flow_png', action='store_true', help='flow_XXXX.pngを保存')
    parser.add_argument('--save_camera_npy', action='store_true', help='camera_field_XXXX.npyを保存')
    parser.add_argument('--save_camera_png', action='store_true', help='camera_field_XXXX.pngを保存')
    parser.add_argument('--save_local_npy', action='store_true', help='local_XXXX.npyを保存')
    parser.add_argument('--save_local_png', action='store_true', help='local_XXXX.pngを保存')
    # 既存出力のスキップ
    parser.add_argument('--skip_existing', action='store_true', help='全ての出力が既に存在する場合は計算をスキップ')
    parser.add_argument('--skip_log_interval', type=int, default=100,
                        help='--skip_existing時、Nペアごとにスキップ数をログ（0で無効）')
    parser.add_argument('--log_missing', action='store_true',
                        help='ペアがスキップされない場合、どの出力が未作成か表示')
    parser.add_argument('--debug_missing_paths', action='store_true',
                        help='--log_missing時、各未作成出力の新旧パスの存在状況も表示')

    # カメラ補間方式＆ローカル閾値（論文方式）
    parser.add_argument('--camera_interp', choices=['stagewise', 'bilinear'], default='stagewise',
                        help='カメラ場の補間方式: stagewise（四隅→辺→全体）またはbilinear（直接）')
    parser.add_argument('--local_thresh', type=float, default=1.0,
                        help='ローカルフローの大きさがこの値未満なら0にする（ピクセル単位、0で無効）')
    args = parser.parse_args()

    # 入力チェック
    if not args.input_dir and not args.video:
        parser.error('input_dirまたはvideoのいずれかを指定してください')

    output_root = os.path.abspath(args.output_root)
    os.makedirs(output_root, exist_ok=True)

    device = get_device()
    setup_gpu_optimizations()
    
    # RAFTモデル構築（単一GPU用にDataParallel除去）
    model = RAFT(args)
    state = torch.load(args.model, map_location=device)
    
    # DataParallel有無でcheckpointのキーを調整
    if 'module.' in next(iter(state.keys())):
        # DataParallelの場合は' module.'を除去
        state = {k.replace('module.', ''): v for k, v in state.items()}
    
    model.load_state_dict(state)
    model.to(device)
    
    # mixed precision有効化
    if args.mixed_precision:
        model = model.half()
    
    model.eval()

    # 動画リスト作成
    videos: List[str]
    if args.input_dir:
        videos = list_videos(args.input_dir, recursive=args.recursive)
    else:
        videos = [args.video]
    if not videos:
        print(f"[INFO] 動画が見つかりません: {args.input_dir or os.path.dirname(args.video)}")
        sys.exit(0)

    # 動画ごとの進捗表示
    vid_iter = videos
    if tqdm is not None:
        vid_iter = tqdm(videos, desc='Videos', unit='vid')

    for vp in vid_iter:
        try:
            ok, msg = process_one_video(args, model, device, vp, output_root)
            if not ok:
                print(msg)
        except Exception as e:
            # エラー時は次の動画へ
            print(f"[ERROR] {vp}: {e}")
            continue

    print(f"完了。出力先: {output_root}（動画ごとに flow/, camera/, local/）")
    # Progress over videos
    vid_iter = videos
    if tqdm is not None:
        vid_iter = tqdm(videos, desc='Videos', unit='vid')

    for vp in vid_iter:
        try:
            ok, msg = process_one_video(args, model, device, vp, output_root)
            if not ok:
                print(msg)
        except Exception as e:
            # Continue with next video on error
            print(f"[ERROR] {vp}: {e}")
            continue

    print(f"Done. Outputs under: {output_root} (per video: flow/, camera/, local/)")
