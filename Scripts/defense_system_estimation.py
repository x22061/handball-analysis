'''
ディフェンスシステムを自動推定するためのスクリプト。
python Scripts/defense_system_estimation.py

入力: data/transformed_player_points.csv
  - columns: frame_num, id, team_color, x, y, direction
  - 値の想定: team_color は 'red' or 'white'、direction は 'left' or 'right'

処理概要:
  1) direction に基づき守備チームを特定（right→red 守備、left→white 守備）。
  2) 各フレームで守備選手の (x, y) が「9mラインの外側」かを判定し、その人数をカウント。
  3) 外側人数に応じてディフェンスシステムを1フレーム単位で推定（0→0_6, 1→1_5, 2→2_4, 3以上→3_3）。
  4) direction の切り替わりで守備フェーズ境界とするが、7秒(=175フレーム)未満の切り替わりはノイズとして無視。
  5) 各フェーズに対して
	  - 最初の7秒(最大175フレーム)における多数決システム
	  - フェーズ全体における多数決システム
	 を算出。

出力: data/defense_system_phases.csv
  - columns: start_frame, end_frame, system_first_7s, system_full

実装上の前提/仮定（不明点に対する合理的な仮定）:
  - 同一フレーム内の direction は基本的に一致していると仮定。万一複数値が混在した場合は多数決（同数は先に現れた値を優先）。
  - 外側人数が3を超える場合は 3_3 に丸める（最大前線人数は3人想定）。
	- 9mラインは以下の近似に基づく（yはコート縦方向0〜1、xは左右0〜1）:
			left 守備側: 中心(0,0.425),(0,0.575) それぞれ半径0.45 の右上/右下の1/4円と，
										それらの円の右端 (x=0.45, y∈[0.425,0.575]) を結ぶ直線 x=0.45。
				  判定は x > x_boundary(y) を「外側」。
	  right 守備側: 上記の左右反転で，直線は x=0.55。判定は x < x_boundary(y) を「外側」。
  - フェーズの最初/最後がデータ端にかかる場合などで7秒未満しか存在しない時は、存在するフレームで多数決。

必要に応じて定数を調整してください。
'''

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


# ===================== 設定値（必要に応じて調整） =====================
FPS = 25  # フレームレート（固定値）
FIRST_WINDOW_SECONDS = 1
FIRST_WINDOW_FRAMES = FIRST_WINDOW_SECONDS * FPS  # 7秒=175フレーム

# 「direction」の一時的切り替わり（ノイズ）と見なす最小継続フレーム数
# 仕様上「守備フェーズは最低7秒間続く」とあるため、ここでは175フレームを閾値とする
MIN_STABLE_FRAMES_FOR_DIRECTION_SWITCH = FIRST_WINDOW_FRAMES


# ===================== データ構造 =====================
@dataclass
class Row:
	frame_num: int
	player_id: str
	team_color: str  # 'red' or 'white'
	x: float
	y: float
	direction: str   # 'left' or 'right'


@dataclass
class Phase:
	start_frame: int
	end_frame: int
	direction: str  # 'left' or 'right'


# ===================== 入出力パス =====================
HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parent.parent  # Scripts/ の親をルートとみなす
DATA_DIR = PROJECT_ROOT / 'data'
INPUT_CSV = DATA_DIR / 'transformed_player_points_offset.csv'
OUTPUT_CSV = DATA_DIR / 'defense_system_phases_offset.csv'
OUTPUT_COUNTS_CSV = DATA_DIR / 'defense_system_counts_offset.csv'
OUTPUT_COUNTS_FIRST_7S_CSV = DATA_DIR / 'defense_system_counts_first_7s_offset.csv'
OUTPUT_COUNTS_FULL_CSV = DATA_DIR / 'defense_system_counts_full_offset.csv'


# ===================== 9mライン判定（幾何） =====================
def boundary_x_for_left_defense(y: float) -> float:
	"""left 側ゴールを守る（= white 守備）ときの9m境界 x 値を返す。

	仕様の近似:
	  - y∈[0.425, 0.575] は直線 x=0.45
	- y>0.575 は上側1/4円: 中心(0,0.575), 半径0.45 → x = sqrt(r^2 - (y-0.575)^2)
	- y<0.425 は下側1/4円: 中心(0,0.425), 半径0.45 → x = sqrt(r^2 - (y-0.425)^2)
	"""
	r = 0.45
	if 0.425 <= y <= 0.575:
		return 0.45
	elif y > 0.575:
		dy = y - 0.575
		# 安全のため負値を0にクランプ
		term = max(r * r - dy * dy, 0.0)
		return term ** 0.5  # x>=0
	else:  # y < 0.425
		dy = y - 0.425
		term = max(r * r - dy * dy, 0.0)
		return term ** 0.5


def boundary_x_for_right_defense(y: float) -> float:
	"""right 側ゴールを守る（= red 守備）ときの9m境界 x 値を返す（左右反転）。

	仕様の近似:
	  - y∈[0.425, 0.575] は直線 x=1-0.45=0.55
	- y>0.575 は上側1/4円: 中心(1,0.575), 半径0.45 → x = 1 - sqrt(r^2 - (y-0.575)^2)
	- y<0.425 は下側1/4円: 中心(1,0.425), 半径0.45 → x = 1 - sqrt(r^2 - (y-0.425)^2)
	"""
	r = 0.45
	if 0.425 <= y <= 0.575:
		return 0.55
	elif y > 0.575:
		dy = y - 0.575
		term = max(r * r - dy * dy, 0.0)
		return 1.0 - (term ** 0.5)
	else:  # y < 0.425
		dy = y - 0.425
		term = max(r * r - dy * dy, 0.0)
		return 1.0 - (term ** 0.5)


def is_outside_9m(x: float, y: float, direction: str) -> bool:
	"""(x, y) が守備側から見て9mラインの「外側」にあるかを判定する。

	- direction='left'  のとき: 境界 x_boundary(y) より x が大きければ外側
	- direction='right' のとき: 境界 x_boundary(y) より x が小さければ外側
	"""
	if direction == 'left':
		bx = boundary_x_for_left_defense(y)
		return x > bx
	elif direction == 'right':
		bx = boundary_x_for_right_defense(y)
		return x < bx
	else:
		# 未知の値は外側ではない扱い（必要なら調整）
		return False


# ===================== CSV 読み込み =====================
def load_rows(csv_path: Path) -> List[Row]:
	"""CSV を読み込み、Row のリストに変換する。"""
	rows: List[Row] = []
	with csv_path.open('r', encoding='utf-8-sig', newline='') as f:
		reader = csv.DictReader(f)
		# 想定カラム: frame_num,id,team_color,x,y,direction
		for rec in reader:
			try:
				frame_num = int(rec['frame_num'])
				player_id = str(rec['id'])
				team_color = str(rec['team_color']).strip().lower()
				x = float(rec['x'])
				y = float(rec['y'])
				# direction 値のノイズ（末尾コロン等）を正規化
				direction_raw = str(rec['direction']).strip().lower()
				direction = direction_raw.rstrip(':')  # 'left:' → 'left'
				if direction not in ('left', 'right'):
					# 既知以外はスキップ
					continue
			except Exception:
				# 欠損やパース失敗はスキップ
				continue
			rows.append(Row(frame_num, player_id, team_color, x, y, direction))
	return rows


# ===================== 方向（direction）をフレーム単位へ集約 =====================
def majority_direction(frame_rows: List[Row]) -> str:
	"""同一フレーム内に複数 direction が混在した場合の多数決（同数は先頭優先）。"""
	counts = Counter(r.direction for r in frame_rows)
	if not counts:
		return 'left'  # デフォルト（到達しない想定）
	# 同数時は登場順優先
	most_common = counts.most_common()
	if len(most_common) >= 2 and most_common[0][1] == most_common[1][1]:
		first_dir = next((r.direction for r in frame_rows if r.direction in (most_common[0][0], most_common[1][0])), most_common[0][0])
		return first_dir
	return most_common[0][0]


# ===================== 守備選手抽出と外側人数カウント =====================
def defensive_team_color(direction: str) -> str:
	"""direction に対応する守備チーム色を返す。"""
	return 'red' if direction == 'right' else 'white'


def count_defenders_outside(frame_rows: List[Row], direction: str) -> int:
	"""そのフレームにおける9m外の守備選手数を数える。"""
	def_color = defensive_team_color(direction)
	cnt = 0
	for r in frame_rows:
		if r.team_color != def_color:
			continue
		if is_outside_9m(r.x, r.y, direction):
			cnt += 1
	return cnt


def map_count_to_system(cnt: int) -> str:
	"""外側人数→ディフェンスシステム表記へ。3以上は3_3に丸める。"""
	if cnt <= 0:
		return '0_6'
	elif cnt == 1:
		return '1_5'
	elif cnt == 2:
		return '2_4'
	else:
		return '3_3'


# ===================== フレーム系列の構築 =====================
def build_frame_series(rows: List[Row]) -> Tuple[List[int], List[str], List[str]]:
	"""フレームごとに direction とフレーム別システムを算出する。

	Returns:
		frames: フレーム番号の昇順リスト
		directions: 各フレームの direction
		systems: 各フレームのディフェンスシステム
	"""
	by_frame: Dict[int, List[Row]] = defaultdict(list)
	for r in rows:
		by_frame[r.frame_num].append(r)

	frames = sorted(by_frame.keys())
	directions: List[str] = []
	systems: List[str] = []

	for fr in frames:
		fr_rows = by_frame[fr]
		d = majority_direction(fr_rows)
		directions.append(d)
		cnt = count_defenders_outside(fr_rows, d)
		systems.append(map_count_to_system(cnt))

	return frames, directions, systems


# ===================== direction 切り替わりのノイズ除去とフェーズ抽出 =====================
def extract_phases(frames: List[int], directions: List[str]) -> List[Phase]:
	"""direction の短時間切り替えを無視し、安定区間をフェーズとして抽出する。

	アルゴリズム（オフライン決定）:
	  - 現在の方向 current_dir を保持。
	  - 別方向が出現したら candidate_run として長さを測定。
	  - candidate_run が MIN_STABLE_FRAMES_FOR_DIRECTION_SWITCH に達した時点で、
		切り替えを確定し、フェーズ境界を candidate_run の開始フレームに置く。
	  - 途中で元の方向に戻ってしまう等で candidate_run が途切れたらノイズとして破棄。
	"""
	if not frames:
		return []

	phases: List[Phase] = []
	current_dir = directions[0]
	phase_start = frames[0]
	candidate_dir = None  # type: ignore
	candidate_start_idx = None  # type: ignore
	candidate_len = 0  # 観測フレーム数（参考）
	candidate_elapsed_frames = 0  # 実フレーム番号ベースの経過

	for i in range(1, len(frames)):
		d = directions[i]
		if candidate_dir is None:
			if d != current_dir:
				candidate_dir = d
				candidate_start_idx = i
				candidate_len = 1
				candidate_elapsed_frames = 0  # 開始時点では0（同一フレーム）
			# 同じなら何もしない
		else:
			# 候補継続/破棄判定
			if d == candidate_dir:
				candidate_len += 1
				# 実フレーム番号の差分で安定性を評価
				candidate_elapsed_frames = frames[i] - frames[candidate_start_idx]
				if candidate_elapsed_frames >= MIN_STABLE_FRAMES_FOR_DIRECTION_SWITCH:
					# 切り替え確定: フェーズをクローズ
					boundary_start_idx = candidate_start_idx
					boundary_start_frame = frames[boundary_start_idx]
					phases.append(Phase(start_frame=phase_start, end_frame=frames[boundary_start_idx - 1], direction=current_dir))
					# 新しいフェーズへ
					current_dir = candidate_dir
					phase_start = boundary_start_frame
					candidate_dir = None
					candidate_start_idx = None
					candidate_len = 0
					candidate_elapsed_frames = 0
			else:
				# 元の方向/別の方向に戻った→短時間のノイズとして候補破棄
				candidate_dir = None
				candidate_start_idx = None
				candidate_len = 0
				candidate_elapsed_frames = 0

	# 最終フェーズをクローズ
	phases.append(Phase(start_frame=phase_start, end_frame=frames[-1], direction=current_dir))
	return phases


# ===================== 集計（多数決） =====================
def majority_label(labels: List[str]) -> str:
	"""多数決で最頻値を返す。同数は先に現れた方を優先。"""
	if not labels:
		return ''
	counts = Counter(labels)
	most_common = counts.most_common()
	if len(most_common) >= 2 and most_common[0][1] == most_common[1][1]:
		# 先に現れたラベルを優先
		seen = set()
		for lb in labels:
			if lb in (most_common[0][0], most_common[1][0]) and lb not in seen:
				return lb
		return most_common[0][0]
	return most_common[0][0]


def summarize_phases(
	phases: List[Phase],
	frames: List[int],
	systems: List[str],
) -> Tuple[List[Tuple[int, int, str, str]], List[Tuple[int, int, str, int, int, int, int]], List[Tuple[int, int, str, int, int, int, int]], List[Tuple[int, int, str, int, int, int, int]]]:
	"""各フェーズの要約とシステム数を返す。
	
	Returns:
		summaries: (start_frame, end_frame, 最初7秒のシステム, 全体システム) のリスト
		system_counts_all: (start_frame, end_frame, direction, 0_6数, 1_5数, 2_4数, 3_3数) 全体のリスト
		system_counts_first_7s: 最初7秒のシステム数リスト
		system_counts_full: フェーズ全体のシステム数リスト
	"""
	# frame index への逆引きテーブル
	frame_to_idx: Dict[int, int] = {fr: i for i, fr in enumerate(frames)}
	results: List[Tuple[int, int, str, str]] = []
	counts_all_results: List[Tuple[int, int, str, int, int, int, int]] = []
	counts_first_7s_results: List[Tuple[int, int, str, int, int, int, int]] = []
	counts_full_results: List[Tuple[int, int, str, int, int, int, int]] = []

	for ph in phases:
		s_idx = frame_to_idx.get(ph.start_frame)
		e_idx = frame_to_idx.get(ph.end_frame)
		if s_idx is None or e_idx is None or s_idx > e_idx:
			continue
		seg_systems = systems[s_idx : e_idx + 1]
		
		# 最初7秒（最大 FIRST_WINDOW_FRAMES）
		first_window_systems = seg_systems[:FIRST_WINDOW_FRAMES]
		system_first = majority_label(first_window_systems)
		system_full = majority_label(seg_systems)
		results.append((ph.start_frame, ph.end_frame, system_first, system_full))
		
		# 全体のシステム数をカウント
		system_counter_all = Counter(seg_systems)
		count_0_6_all = system_counter_all.get('0_6', 0)
		count_1_5_all = system_counter_all.get('1_5', 0)
		count_2_4_all = system_counter_all.get('2_4', 0)
		count_3_3_all = system_counter_all.get('3_3', 0)
		
		counts_all_results.append((ph.start_frame, ph.end_frame, ph.direction, 
		                          count_0_6_all, count_1_5_all, count_2_4_all, count_3_3_all))
		
		# 最初7秒のシステム数をカウント
		system_counter_first_7s = Counter(first_window_systems)
		count_0_6_first = system_counter_first_7s.get('0_6', 0)
		count_1_5_first = system_counter_first_7s.get('1_5', 0)
		count_2_4_first = system_counter_first_7s.get('2_4', 0)
		count_3_3_first = system_counter_first_7s.get('3_3', 0)
		
		counts_first_7s_results.append((ph.start_frame, ph.end_frame, ph.direction,
		                               count_0_6_first, count_1_5_first, count_2_4_first, count_3_3_first))
		
		# フェーズ全体のシステム数（全体と同じだが、明示的に分離）
		counts_full_results.append((ph.start_frame, ph.end_frame, ph.direction,
		                           count_0_6_all, count_1_5_all, count_2_4_all, count_3_3_all))

	return results, counts_all_results, counts_first_7s_results, counts_full_results


# ===================== メイン処理 =====================
def main(input_csv: Path = INPUT_CSV, output_csv: Path = OUTPUT_CSV, 
         output_counts_csv: Path = OUTPUT_COUNTS_CSV, 
         output_counts_first_7s_csv: Path = OUTPUT_COUNTS_FIRST_7S_CSV,
         output_counts_full_csv: Path = OUTPUT_COUNTS_FULL_CSV) -> None:
	# 入力チェック
	if not input_csv.exists():
		raise FileNotFoundError(f"入力CSVが見つかりません: {input_csv}")

	# 1) CSV読み込み
	rows = load_rows(input_csv)
	if not rows:
		raise RuntimeError("CSVに有効な行がありませんでした。カラム名や内容をご確認ください。")

	# 2) フレーム系列（direction, system）を構築
	frames, directions, systems = build_frame_series(rows)

	# 3) フェーズ抽出（direction ノイズ除去）
	phases = extract_phases(frames, directions)

	# 4) 各フェーズを要約
	summaries, system_counts_all, system_counts_first_7s, system_counts_full = summarize_phases(phases, frames, systems)

	# 5) CSV 出力（従来の出力）
	with output_csv.open('w', encoding='utf-8', newline='') as f:
		writer = csv.writer(f)
		writer.writerow(['start_frame', 'end_frame', 'system_first_7s', 'system_full'])
		for s_frame, e_frame, sys_first, sys_full in summaries:
			writer.writerow([s_frame, e_frame, sys_first, sys_full])

	# 6) システム数CSV出力（全体）
	with output_counts_csv.open('w', encoding='utf-8', newline='') as f:
		writer = csv.writer(f)
		writer.writerow(['start_frame', 'end_frame', 'direction', '0_6', '1_5', '2_4', '3_3'])
		for s_frame, e_frame, direction, count_0_6, count_1_5, count_2_4, count_3_3 in system_counts_all:
			writer.writerow([s_frame, e_frame, direction, count_0_6, count_1_5, count_2_4, count_3_3])

	# 7) システム数CSV出力（最初7秒）
	with output_counts_first_7s_csv.open('w', encoding='utf-8', newline='') as f:
		writer = csv.writer(f)
		writer.writerow(['start_frame', 'end_frame', 'direction', '0_6', '1_5', '2_4', '3_3'])
		for s_frame, e_frame, direction, count_0_6, count_1_5, count_2_4, count_3_3 in system_counts_first_7s:
			writer.writerow([s_frame, e_frame, direction, count_0_6, count_1_5, count_2_4, count_3_3])

	# 8) システム数CSV出力（フェーズ全体）
	with output_counts_full_csv.open('w', encoding='utf-8', newline='') as f:
		writer = csv.writer(f)
		writer.writerow(['start_frame', 'end_frame', 'direction', '0_6', '1_5', '2_4', '3_3'])
		for s_frame, e_frame, direction, count_0_6, count_1_5, count_2_4, count_3_3 in system_counts_full:
			writer.writerow([s_frame, e_frame, direction, count_0_6, count_1_5, count_2_4, count_3_3])

	# 簡易ログ
	print(f"フェーズ数: {len(summaries)} を {output_csv} に出力しました。")
	print(f"システム数集計（全体）: {len(system_counts_all)} フェーズを {output_counts_csv} に出力しました。")
	print(f"システム数集計（最初7秒）: {len(system_counts_first_7s)} フェーズを {output_counts_first_7s_csv} に出力しました。")
	print(f"システム数集計（フェーズ全体）: {len(system_counts_full)} フェーズを {output_counts_full_csv} に出力しました。")


if __name__ == '__main__':
	main()

