import pandas as pd
import numpy as np
from collections import Counter
import math

def calculate_distance(x1, y1, x2, y2):
    """2点間の距離を計算"""
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def point_in_polygon(x, y, polygon):
    """点が多角形内にあるかを判定"""
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def get_zone_left(x, y):
    """LEFT攻撃時のゾーン判定"""
    # ゾーン1: x>=0.45, y>0.575
    if x >= 0.45 and y > 0.575:
        return 1
    
    # ゾーン2: x>=0.45, 0.425<=y<=0.575  
    if x >= 0.45 and 0.425 <= y <= 0.575:
        return 2
    
    # ゾーン3: x>=0.45, y<0.425
    if x >= 0.45 and y < 0.425:
        return 3
    
    # 拡張ゾーン4: x≤0からの範囲を含む
    # 実際の座標で定義: (-1, -1), (0.45, -1), (0.15, 0.275), (-1, 0.275)
    zone4 = [(-1, -1), (0.45, -1), (0.15, 0.275), (-1, 0.275)]
    if point_in_polygon(x, y, zone4):
        return 4
    
    # 拡張ゾーン5: x≤0からの範囲を含む  
    # 実際の座標で定義: (-1, 0.275), (0.15, 0.275), (0.45, 0), (0.45, 0.425), (-1, 0.425)
    zone5 = [(-1, 0.275), (0.15, 0.275), (0.45, 0), (0.45, 0.425), (-1, 0.425)]
    if point_in_polygon(x, y, zone5):
        return 5
    
    # 拡張ゾーン6: x≤0からの範囲を含む
    # 実際の座標で定義: (-1, 0.425), (0.45, 0.425), (0.45, 0.575), (-1, 0.575)
    zone6 = [(-1, 0.425), (0.45, 0.425), (0.45, 0.575), (-1, 0.575)]
    if point_in_polygon(x, y, zone6):
        return 6
    
    # 拡張ゾーン7: x≤0からの範囲を含む
    # 実際の座標で定義: (-1, 0.575), (0.45, 0.575), (0.45, 1), (0.15, 0.725), (-1, 0.725)
    zone7 = [(-1, 0.575), (0.45, 0.575), (0.45, 1), (0.15, 0.725), (-1, 0.725)]
    if point_in_polygon(x, y, zone7):
        return 7
    
    # 拡張ゾーン8: x≤0とy≥1の範囲を含む
    # 実際の座標で定義: (-1, 0.725), (0.15, 0.725), (0.45, 2), (-1, 2)
    zone8 = [(-1, 0.725), (0.15, 0.725), (0.45, 2), (-1, 2)]
    if point_in_polygon(x, y, zone8):
        return 8
    
    # 最後の手段として、どのゾーンにも入らない場合は最も近いゾーンを返す
    # この実装では、まずy座標で大まかに分類
    if y < 0.275:
        return 4
    elif y < 0.425:
        return 5
    elif y < 0.575:
        return 6
    elif y < 0.725:
        return 7
    else:
        return 8

def get_zone(x, y, direction):
    """方向に応じてゾーンを判定"""
    if direction == 'left':
        return get_zone_left(x, y)
    else:  # right
        # x座標を反転してLEFT基準で判定
        x_flipped = 1 - x
        return get_zone_left(x_flipped, y)

def calculate_player_score(player_data, direction):
    """選手のスコアを計算（連続ゴール方向移動中の速度 + 連続ゴール方向移動量）"""
    if len(player_data) < 2:
        return 0, None, None
    
    # フレーム順にソート
    player_data = player_data.sort_values('frame_num').reset_index(drop=True)
    
    # 連続ゴール方向移動の最長区間を見つける
    max_goal_movement = 0
    max_goal_speed = 0
    max_goal_final_pos = None  # 最大連続移動区間の最終座標
    
    current_goal_movement = 0
    current_goal_speeds = []
    current_goal_start_idx = None
    current_goal_end_idx = None
    
    for i in range(1, len(player_data)):
        x_prev, x_curr = player_data.iloc[i-1]['x'], player_data.iloc[i]['x']
        y_prev, y_curr = player_data.iloc[i-1]['y'], player_data.iloc[i]['y']
        
        # フレーム間距離と時間差
        dist = calculate_distance(x_prev, y_prev, x_curr, y_curr)
        time_diff = player_data.iloc[i]['frame_num'] - player_data.iloc[i-1]['frame_num']
        
        # ゴール方向への移動かチェック
        is_goal_direction = False
        if direction == 'left' and x_curr < x_prev:  # x=0方向
            is_goal_direction = True
        elif direction == 'right' and x_curr > x_prev:  # x=1方向
            is_goal_direction = True
        
        if is_goal_direction:
            # ゴール方向への移動
            if current_goal_start_idx is None:
                current_goal_start_idx = i-1
            current_goal_end_idx = i
            current_goal_movement += dist
            if time_diff > 0:
                current_goal_speeds.append(dist / time_diff)
        else:
            # ゴール方向でない場合、現在の連続移動が最大かチェック
            if current_goal_movement > max_goal_movement:
                max_goal_movement = current_goal_movement
                max_goal_speed = np.mean(current_goal_speeds) if current_goal_speeds else 0
                # 最大連続移動区間の最終座標を記録
                if current_goal_end_idx is not None:
                    max_goal_final_pos = (
                        player_data.iloc[current_goal_end_idx]['x'],
                        player_data.iloc[current_goal_end_idx]['y']
                    )
            
            # リセット
            current_goal_movement = 0
            current_goal_speeds = []
            current_goal_start_idx = None
            current_goal_end_idx = None
    
    # 最後の連続移動もチェック
    if current_goal_movement > max_goal_movement:
        max_goal_movement = current_goal_movement
        max_goal_speed = np.mean(current_goal_speeds) if current_goal_speeds else 0
        # 最大連続移動区間の最終座標を記録
        if current_goal_end_idx is not None:
            max_goal_final_pos = (
                player_data.iloc[current_goal_end_idx]['x'],
                player_data.iloc[current_goal_end_idx]['y']
            )
    
    # スコア = 連続ゴール方向移動中の平均速度 + 最大連続ゴール方向移動量
    score = max_goal_speed + max_goal_movement
    
    # 最大連続ゴール方向移動の最終位置、なければ全体の最終位置
    if max_goal_final_pos is not None:
        final_x, final_y = max_goal_final_pos
    else:
        final_x = player_data.iloc[-1]['x']
        final_y = player_data.iloc[-1]['y']
    
    return score, final_x, final_y

def process_phase(phase_data, frame_range_start, frame_range_end):
    """1つのフェーズを処理"""
    # フレーム範囲内のデータを取得
    phase_frames = phase_data[
        (phase_data['frame_num'] >= frame_range_start) & 
        (phase_data['frame_num'] <= frame_range_end)
    ].copy()
    
    if len(phase_frames) == 0:
        return 'no_data', None
    
    # 方向を多数決で決定
    direction_counts = Counter(phase_frames['direction'])
    direction = direction_counts.most_common(1)[0][0]
    
    # 攻撃選手の特定
    if direction == 'left':
        attack_players = phase_frames[phase_frames['team_color'] == 'white']
    else:  # right
        attack_players = phase_frames[phase_frames['team_color'] == 'red']
    
    if len(attack_players) == 0:
        return 'no_data', None
    
    # 各攻撃選手のスコアを計算
    best_score = -1
    best_player_final_pos = None
    best_player_id = None
    
    for player_id in attack_players['id'].unique():
        player_data = attack_players[attack_players['id'] == player_id]
        score, final_x, final_y = calculate_player_score(player_data, direction)
        
        if score > best_score:
            best_score = score
            best_player_final_pos = (final_x, final_y)
            best_player_id = player_id
    
    if best_player_final_pos is None:
        return 'no_data', None
    
    # ゾーン判定
    zone = get_zone(best_player_final_pos[0], best_player_final_pos[1], direction)
    
    return str(zone), best_player_id

def main():
    # データ読み込み
    frames_df = pd.read_csv('data/defense_system_counts_offset_500frames.csv')
    player_df = pd.read_csv('data/transformed_player_points_offset.csv')
    
    results = []
    
    for _, row in frames_df.iterrows():
        start_frame = row['start_frame']
        end_frame = row['end_frame']
        
        zone, player_id = process_phase(player_df, start_frame, end_frame)
        
        results.append({
            'start_frame': start_frame,
            'end_frame': end_frame,
            'zone': zone,
            'player_id': player_id if player_id is not None else 'no_data'
        })
    
    # 結果を保存
    result_df = pd.DataFrame(results)
    result_df.to_csv('data/shoot_position_results.csv', index=False)
    print(f"Results saved to data/shoot_position_results.csv")
    print(f"Processed {len(results)} phases")

if __name__ == "__main__":
    main()
