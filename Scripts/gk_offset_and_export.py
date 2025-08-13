"""
GKオフセット及びエクスポート処理スクリプト。
python Scripts/gk_offset_and_export.py

入力: data/transformed_player_points.csv
  - columns: frame_num, id, team_color, x, y, direction

処理概要:
  1) defense_system_estimation.pyと同様の方法で守備フェーズを推定。
  2) 各守備フェーズで、ゴール座標に最も近い選手をGKとして特定。
     - left守備(white): (0.05, 0.5)に最も近い選手
     - right守備(red): (0.95, 0.5)に最も近い選手
  3) GKの座標を目標座標に移動させるためのオフセットを計算し、
     そのフェーズの全選手に同じオフセットを適用。
  4) 欠損フレームのデータを線形補完で補う。
  5) is_gkカラムを追加（GK=1、その他=0）。

出力: data/transformed_player_points_offset.csv
  - columns: frame_num, id, team_color, x, y, direction, is_gk
"""

from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


# ===================== 設定値 =====================
FPS = 25
FIRST_WINDOW_SECONDS = 7
FIRST_WINDOW_FRAMES = FIRST_WINDOW_SECONDS * FPS
MIN_STABLE_FRAMES_FOR_DIRECTION_SWITCH = FIRST_WINDOW_FRAMES

# ゴール座標
GOAL_LEFT = (0.05, 0.5)   # left守備(white)のゴール
GOAL_RIGHT = (0.95, 0.5)  # right守備(red)のゴール


# ===================== データ構造 =====================
@dataclass
class Row:
    frame_num: int
    player_id: str
    team_color: str
    x: float
    y: float
    direction: str


@dataclass
class Phase:
    start_frame: int
    end_frame: int
    direction: str


@dataclass
class OutputRow:
    frame_num: int
    player_id: str
    team_color: str
    x: float
    y: float
    direction: str
    is_gk: int


# ===================== 入出力パス =====================
HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
INPUT_CSV = DATA_DIR / 'transformed_player_points.csv'
OUTPUT_CSV = DATA_DIR / 'transformed_player_points_offset.csv'
PHASE_SUMMARY_CSV = DATA_DIR / 'gk_phase_summary.csv'


# ===================== ユーティリティ関数 =====================
def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """2点間の距離を計算。"""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def defensive_team_color(direction: str) -> str:
    """direction に対応する守備チーム色を返す。"""
    return 'red' if direction == 'right' else 'white'


def get_goal_position(direction: str) -> Tuple[float, float]:
    """direction に対応するゴール座標を返す。"""
    return GOAL_RIGHT if direction == 'right' else GOAL_LEFT


# ===================== CSV 読み込み =====================
def load_rows(csv_path: Path) -> List[Row]:
    """CSV を読み込み、Row のリストに変換する。"""
    rows: List[Row] = []
    with csv_path.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        for rec in reader:
            try:
                frame_num = int(rec['frame_num'])
                player_id = str(rec['id'])
                team_color = str(rec['team_color']).strip().lower()
                x = float(rec['x'])
                y = float(rec['y'])
                direction_raw = str(rec['direction']).strip().lower()
                direction = direction_raw.rstrip(':')
                if direction not in ('left', 'right'):
                    continue
            except Exception:
                continue
            rows.append(Row(frame_num, player_id, team_color, x, y, direction))
    return rows


# ===================== フレーム系列の構築 =====================
def majority_direction(frame_rows: List[Row]) -> str:
    """同一フレーム内に複数 direction が混在した場合の多数決。"""
    counts = Counter(r.direction for r in frame_rows)
    if not counts:
        return 'left'
    most_common = counts.most_common()
    if len(most_common) >= 2 and most_common[0][1] == most_common[1][1]:
        first_dir = next((r.direction for r in frame_rows 
                         if r.direction in (most_common[0][0], most_common[1][0])), 
                        most_common[0][0])
        return first_dir
    return most_common[0][0]


def build_frame_series(rows: List[Row]) -> Tuple[List[int], List[str]]:
    """フレームごとに direction を算出する。"""
    by_frame: Dict[int, List[Row]] = defaultdict(list)
    for r in rows:
        by_frame[r.frame_num].append(r)

    frames = sorted(by_frame.keys())
    directions: List[str] = []

    for fr in frames:
        fr_rows = by_frame[fr]
        d = majority_direction(fr_rows)
        directions.append(d)

    return frames, directions


# ===================== フェーズ抽出 =====================
def extract_phases(frames: List[int], directions: List[str]) -> List[Phase]:
    """direction の短時間切り替えを無視し、安定区間をフェーズとして抽出する。"""
    if not frames:
        return []

    phases: List[Phase] = []
    current_dir = directions[0]
    phase_start = frames[0]
    candidate_dir = None
    candidate_start_idx = None
    candidate_elapsed_frames = 0

    for i in range(1, len(frames)):
        d = directions[i]
        if candidate_dir is None:
            if d != current_dir:
                candidate_dir = d
                candidate_start_idx = i
                candidate_elapsed_frames = 0
        else:
            if d == candidate_dir:
                candidate_elapsed_frames = frames[i] - frames[candidate_start_idx]
                if candidate_elapsed_frames >= MIN_STABLE_FRAMES_FOR_DIRECTION_SWITCH:
                    boundary_start_frame = frames[candidate_start_idx]
                    phases.append(Phase(start_frame=phase_start, 
                                      end_frame=frames[candidate_start_idx - 1], 
                                      direction=current_dir))
                    current_dir = candidate_dir
                    phase_start = boundary_start_frame
                    candidate_dir = None
                    candidate_start_idx = None
                    candidate_elapsed_frames = 0
            else:
                candidate_dir = None
                candidate_start_idx = None
                candidate_elapsed_frames = 0

    phases.append(Phase(start_frame=phase_start, end_frame=frames[-1], direction=current_dir))
    return phases


# ===================== GK特定とオフセット計算 =====================
def find_gk_for_phase(rows: List[Row], phase: Phase) -> Optional[str]:
    """指定フェーズでGKとなる選手IDを特定する（全選手から選択）。"""
    goal_pos = get_goal_position(phase.direction)
    
    # フェーズ内の全選手を収集
    phase_players = [r for r in rows 
                    if phase.start_frame <= r.frame_num <= phase.end_frame]
    
    if not phase_players:
        return None
    
    # 各選手のゴールからの平均距離を計算
    player_distances: Dict[str, List[float]] = defaultdict(list)
    for r in phase_players:
        dist = distance((r.x, r.y), goal_pos)
        player_distances[r.player_id].append(dist)
    
    # 平均距離が最小の選手をGKとする
    avg_distances = {pid: sum(dists) / len(dists) 
                    for pid, dists in player_distances.items()}
    
    if not avg_distances:
        return None
        
    return min(avg_distances.keys(), key=lambda pid: avg_distances[pid])


def interpolate_gk_positions(rows: List[Row], phase: Phase, gk_id: str) -> List[Row]:
    """フェーズ内でGKの欠損座標を線形補完する。"""
    # フェーズ内のGKデータを取得
    gk_rows = [r for r in rows 
               if phase.start_frame <= r.frame_num <= phase.end_frame 
               and r.player_id == gk_id]
    
    if not gk_rows:
        return []
    
    # フレーム順にソート
    gk_rows.sort(key=lambda r: r.frame_num)
    
    # 補完後のデータを格納
    interpolated_rows = []
    
    # 最初のデータポイントを追加
    interpolated_rows.append(gk_rows[0])
    
    # 各連続するデータポイント間で線形補完
    for i in range(1, len(gk_rows)):
        prev_row = gk_rows[i - 1]
        curr_row = gk_rows[i]
        
        # 連続するフレーム間で補完が必要かチェック
        frame_gap = curr_row.frame_num - prev_row.frame_num
        if frame_gap > 1:
            # 補完フレームを生成
            for frame_offset in range(1, frame_gap):
                interp_frame = prev_row.frame_num + frame_offset
                
                # 線形補完
                ratio = frame_offset / frame_gap
                interp_x = prev_row.x + (curr_row.x - prev_row.x) * ratio
                interp_y = prev_row.y + (curr_row.y - prev_row.y) * ratio
                
                interp_row = Row(
                    frame_num=interp_frame,
                    player_id=gk_id,
                    team_color=prev_row.team_color,
                    x=interp_x,
                    y=interp_y,
                    direction=prev_row.direction
                )
                interpolated_rows.append(interp_row)
        
        # 現在のデータポイントを追加
        interpolated_rows.append(curr_row)
    
    return interpolated_rows


def calculate_phase_offset(gk_rows: List[Row], phase: Phase) -> Tuple[float, float]:
    """フェーズ内のGK座標の平均からオフセットを計算する。"""
    if not gk_rows:
        return (0.0, 0.0)
    
    goal_pos = get_goal_position(phase.direction)
    
    # GKの平均座標を計算
    avg_x = sum(r.x for r in gk_rows) / len(gk_rows)
    avg_y = sum(r.y for r in gk_rows) / len(gk_rows)
    
    # オフセット計算
    offset_x = goal_pos[0] - avg_x
    offset_y = goal_pos[1] - avg_y
    
    return (offset_x, offset_y)



# ===================== 線形補完 =====================
def interpolate_missing_frames(rows: List[Row]) -> List[Row]:
    """欠損フレームのデータを線形補完で補う。"""
    # 選手IDごとにデータを整理
    by_player: Dict[str, List[Row]] = defaultdict(list)
    for r in rows:
        by_player[r.player_id].append(r)
    
    # 各選手のデータをフレーム順にソート
    for pid in by_player:
        by_player[pid].sort(key=lambda r: r.frame_num)
    
    interpolated_rows: List[Row] = []
    
    for pid, player_rows in by_player.items():
        if len(player_rows) < 2:
            interpolated_rows.extend(player_rows)
            continue
        
        current_rows = [player_rows[0]]  # 最初のフレームは必ず含める
        
        for i in range(1, len(player_rows)):
            prev_row = player_rows[i - 1]
            curr_row = player_rows[i]
            
            # 連続するフレーム間で補完が必要かチェック
            frame_gap = curr_row.frame_num - prev_row.frame_num
            if frame_gap > 1:
                # 補完フレームを生成
                for frame_offset in range(1, frame_gap):
                    interp_frame = prev_row.frame_num + frame_offset
                    
                    # 線形補完
                    ratio = frame_offset / frame_gap
                    interp_x = prev_row.x + (curr_row.x - prev_row.x) * ratio
                    interp_y = prev_row.y + (curr_row.y - prev_row.y) * ratio
                    
                    interp_row = Row(
                        frame_num=interp_frame,
                        player_id=pid,
                        team_color=prev_row.team_color,
                        x=interp_x,
                        y=interp_y,
                        direction=prev_row.direction
                    )
                    current_rows.append(interp_row)
            
            current_rows.append(curr_row)
        
        interpolated_rows.extend(current_rows)
    
    return interpolated_rows


# ===================== メイン処理 =====================
def main(input_csv: Path = INPUT_CSV, output_csv: Path = OUTPUT_CSV) -> None:
    # 入力チェック
    if not input_csv.exists():
        raise FileNotFoundError(f"入力CSVが見つかりません: {input_csv}")

    print("CSV読み込み中...")
    rows = load_rows(input_csv)
    if not rows:
        raise RuntimeError("CSVに有効な行がありませんでした。")

    print("欠損フレームの補完中...")
    rows = interpolate_missing_frames(rows)

    print("フェーズ抽出中...")
    frames, directions = build_frame_series(rows)
    phases = extract_phases(frames, directions)

    print(f"{len(phases)}個の守備フェーズを検出しました。")

    # 各フェーズでGKを特定し、座標補完とオフセット計算を実行
    all_gk_rows: List[Row] = []  # 補完されたGKデータ
    phase_gks: Dict[int, str] = {}  # フレーム番号 -> GK ID
    phase_offsets: Dict[int, Tuple[float, float]] = {}  # フレーム番号 -> オフセット
    phase_summaries: List[Tuple[int, int, str, str, float, float]] = []  # フェーズ要約情報
    
    gk_found_count = 0
    no_gk_phases = []

    for i, phase in enumerate(phases):
        print(f"フェーズ {i+1}/{len(phases)} 処理中...")
        
        # フェーズでGKを特定
        gk_id = find_gk_for_phase(rows, phase)
        if gk_id:
            gk_found_count += 1
            
            # GK座標を線形補完
            interpolated_gk_rows = interpolate_gk_positions(rows, phase, gk_id)
            all_gk_rows.extend(interpolated_gk_rows)
            
            # オフセット計算
            offset = calculate_phase_offset(interpolated_gk_rows, phase)
            
            # フェーズ内の全フレームにGK IDとオフセットを設定
            for frame_num in range(phase.start_frame, phase.end_frame + 1):
                phase_gks[frame_num] = gk_id
                phase_offsets[frame_num] = offset
            
            # フェーズ要約情報を記録
            phase_summaries.append((
                phase.start_frame, phase.end_frame, phase.direction, 
                gk_id, offset[0], offset[1]
            ))
                
            print(f"  GK: {gk_id}, 補完データ数: {len(interpolated_gk_rows)}, オフセット: ({offset[0]:.3f}, {offset[1]:.3f})")
        else:
            no_gk_phases.append(i+1)
            print(f"  警告: フェーズ {i+1} (フレーム {phase.start_frame}-{phase.end_frame}) でGKが特定できませんでした")
            # オフセットなしで処理を続行
            for frame_num in range(phase.start_frame, phase.end_frame + 1):
                phase_offsets[frame_num] = (0.0, 0.0)
            
            # フェーズ要約情報を記録（GKなし）
            phase_summaries.append((
                phase.start_frame, phase.end_frame, phase.direction, 
                "", 0.0, 0.0
            ))

    print(f"GKが特定されたフェーズ: {gk_found_count}/{len(phases)}")
    if no_gk_phases:
        print(f"GKが特定されなかったフェーズ: {no_gk_phases}")
    print(f"補完されたGKデータ総数: {len(all_gk_rows)}")

    print("オフセット適用とCSV出力中...")
    
    # 元データと補完されたGKデータをマージ（重複除去）
    # 元データを優先し、補完データは欠損部分のみ追加
    existing_keys = set((r.frame_num, r.player_id) for r in rows)
    additional_gk_rows = [r for r in all_gk_rows if (r.frame_num, r.player_id) not in existing_keys]
    all_rows = rows + additional_gk_rows
    
    print(f"元データ: {len(rows)}行, 追加GKデータ: {len(additional_gk_rows)}行")
    
    # デバッグ: 設定されたGKの情報を確認
    print(f"GKが設定されたフレーム数: {len(phase_gks)}")
    if phase_gks:
        sample_frames = list(phase_gks.keys())[:5]
        print(f"サンプルフレーム: {sample_frames}")
        for frame in sample_frames:
            print(f"  フレーム {frame}: GK ID = {phase_gks[frame]}")
    
    # 出力データ生成
    output_rows: List[OutputRow] = []
    gk_assignment_count = 0
    
    for r in all_rows:
        # オフセット適用
        offset = phase_offsets.get(r.frame_num, (0.0, 0.0))
        new_x = r.x + offset[0]
        new_y = r.y + offset[1]
        
        # GK判定: そのフレームでGKとして設定された選手IDと一致するかチェック
        frame_gk_id = phase_gks.get(r.frame_num)
        is_gk = 1 if frame_gk_id == r.player_id else 0
        if is_gk == 1:
            gk_assignment_count += 1
        
        output_rows.append(OutputRow(
            frame_num=r.frame_num,
            player_id=r.player_id,
            team_color=r.team_color,
            x=new_x,
            y=new_y,
            direction=r.direction,
            is_gk=is_gk
        ))
    
    print(f"GKとして設定された選手データ数: {gk_assignment_count}")
    print(f"全データ数: {len(output_rows)}")

    # CSV出力
    with output_csv.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame_num', 'id', 'team_color', 'x', 'y', 'direction', 'is_gk'])
        for row in output_rows:
            writer.writerow([
                row.frame_num, row.player_id, row.team_color, 
                row.x, row.y, row.direction, row.is_gk
            ])

    # フェーズ要約CSV出力
    with PHASE_SUMMARY_CSV.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['start_frame', 'end_frame', 'direction', 'gk_id', 'offset_x', 'offset_y'])
        for summary in phase_summaries:
            writer.writerow(summary)

    print(f"処理完了:")
    print(f"  選手データ: {len(output_rows)}行を {output_csv} に出力")
    print(f"  フェーズ要約: {len(phase_summaries)}行を {PHASE_SUMMARY_CSV} に出力")


if __name__ == '__main__':
    main()

