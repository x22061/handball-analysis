'''
映像を112x112のRGBフレームに分解して保存するスクリプト
C3Dの入力に合わせてリサイズとパディングを行う
'''

import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

def extract_and_save_frames(input_dir, output_dir, target_width=112):
    os.makedirs(output_dir, exist_ok=True)
    mp4_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.mp4')]
    print(f"Found {len(mp4_files)} mp4 files.")
    for idx, mp4_file in enumerate(mp4_files):
        video_path = os.path.join(input_dir, mp4_file)
        base_name = os.path.splitext(mp4_file)[0]
        out_folder = os.path.join(output_dir, base_name)
        os.makedirs(out_folder, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[{idx+1}/{len(mp4_files)}] Processing {mp4_file} ({total_frames} frames)...")
        frame_idx = 0
        with tqdm(total=total_frames, desc=f"{base_name}", unit="frame") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, _ = frame.shape
                scale = target_width / w
                new_w = target_width
                new_h = int(h * scale)
                resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                # 黒埋めパディング
                pad_h = 112
                pad = np.zeros((pad_h, new_w, 3), dtype=np.uint8)
                y_offset = (pad_h - new_h) // 2
                pad[y_offset:y_offset+new_h, :, :] = resized
                # 0-1正規化
                norm = pad.astype(np.float32) / 255.0
                # 保存
                out_path = os.path.join(out_folder, f"rgb_frame_{frame_idx:04d}.png")
                img = Image.fromarray((norm * 255).astype(np.uint8))
                img.save(out_path)
                frame_idx += 1
                pbar.update(1)
        cap.release()
    print("All videos processed.")

if __name__ == "__main__":
    extract_and_save_frames(
        input_dir=r"D:\data\video",
        output_dir=r"D:\data\video\RGB"
    )
