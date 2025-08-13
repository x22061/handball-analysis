'''
守備フェーズが7秒を超えなかったものを除外するコード
'''
# import pandas as pd

# # 入力ファイル
# input_csv = 'data/defense_system_counts_offset.csv'
# # 出力ファイル
# output_csv = 'data/defense_system_counts_offset_filtered.csv'

# # CSV読み込み
# df = pd.read_csv(input_csv)

# # フレーム数計算
# df['frame_count'] = df['end_frame'] - df['start_frame'] + 1

# # 175フレーム以下の行を除外
# df_filtered = df[df['frame_count'] > 175].drop(columns=['frame_count'])

# # 新しいCSVに保存
# df_filtered.to_csv(output_csv, index=False)

# print(f'Filtered CSV saved to: {output_csv}')

'''
シュート推定に使用する範囲のフレームを抽出するコード
'''
import pandas as pd

# 入力CSVファイル
input_csv = 'data/defense_system_counts_offset.csv'
# 出力CSVファイル
output_csv = 'data/defense_system_counts_offset_500frames.csv'

# CSV読み込み
df = pd.read_csv(input_csv)

# 新しいstart_frame, end_frameを計算
df['start_frame'] = df['end_frame'] - 499
df['start_frame'] = df['start_frame'].clip(lower=0)  # 0未満は0に

# 2列だけ抽出
result = df[['start_frame', 'end_frame']]

# 出力
result.to_csv(output_csv, index=False)
print(f"Saved: {output_csv}")