# ハンドボール分析ツールキット

ハンドボールの試合映像を分析するためのスクリプト群。

## 主な機能
- RAFTでflow,camera,localのオプティカルフローを計算する `raft_motion_analysis.py`
- Horn–Schunck法でカメラのオプティカルフローを計算する `hs_pyramid.py`
- GK フェーズ / 守備システム解析 `gk_offset_and_export.py`, `defense_system_estimation.py`
- 選手のポジションデータ作成・編集 `manual_player_plotter.py`, `handball_player_gui.py`

## 環境構築
動かした環境　
Python: 3.13.5 （GPU 機能は CUDA 対応 PyTorch が必要）

PyTorch: 2.6.0+cu124 (RTX4070)

仮想環境作成 / 有効化:  
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
主要ライブラリ（必要に応じ追記）:
```powershell
pip install -r requirements.txt
```

## 主要スクリプト概要

### 1. オプティカルフロー推定 `raft_motion_analysis.py`
全体のオプティカルフローを計算。(flow)
四隅のオプティカルフローを計算。線形補完で画面全体のオプティカルフローを計算(camera)
全体からカメラの動きを引いた差分を計算。(local)

### 2. ディフェンスシステム推定 `defense_system_estimation.py`
ゴール物体の向きから守備フェーズと守備チームを推定。
9mラインの外側にいる守備選手の数をカウント。
フェーズごとに、全体と最初の7秒間のディフェンスシステムを出力。


### 3. 選手の座標データ作成GUI `manual_player_plotter.py`, `handball_player_gui.py` 
`manual_player_plotter.py`: 上面図に選手の位置をプロット。座標データを保存。
`handball_player_gui.py`: 作成したデータを確認。編集・保存も可能。 

