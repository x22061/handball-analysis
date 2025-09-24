'''
選手の上面図上での座標を表示・編集するためのGUIアプリケーション。
python Scripts/handball_player_gui.py
'''

import tkinter as tk
from tkinter import ttk
import pandas as pd
import numpy as np
from PIL import Image, ImageTk
import threading
import time

# ディフェンスシステム推定のための関数（defense_system_estimation.pyから移植）
def boundary_x_for_left_defense(y: float) -> float:
    """left 側ゴールを守る（= white 守備）ときの9m境界 x 値を返す。"""
    r = 0.45
    if 0.425 <= y <= 0.575:
        return 0.45
    elif y > 0.575:
        dy = y - 0.575
        term = max(r * r - dy * dy, 0.0)
        return term ** 0.5
    else:  # y < 0.425
        dy = y - 0.425
        term = max(r * r - dy * dy, 0.0)
        return term ** 0.5

def boundary_x_for_right_defense(y: float) -> float:
    """right 側ゴールを守る（= red 守備）ときの9m境界 x 値を返す（左右反転）。"""
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
    """(x, y) が守備側から見て9mラインの「外側」にあるかを判定する。"""
    if direction == 'left':
        bx = boundary_x_for_left_defense(y)
        return x > bx
    elif direction == 'right':
        bx = boundary_x_for_right_defense(y)
        return x < bx
    else:
        return False

def defensive_team_color(direction: str) -> str:
    """direction に対応する守備チーム色を返す。"""
    return 'red' if direction == 'right' else 'white'

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

class HandballPlayerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ハンドボール選手座標再生GUI")
        self.root.geometry("1000x800")  # さらに縮小
        
        # 表示領域の設定
        self.canvas_width = 800   # キャンバス全体の幅をさらに縮小
        self.canvas_height = 650  # キャンバス全体の高さをさらに縮小
        self.court_width = 550    # コート画像の幅をさらに縮小
        self.court_height = 550   # コート画像の高さをさらに縮小
        # コート画像を中央に配置
        self.court_offset_x = (self.canvas_width - self.court_width) // 2
        self.court_offset_y = (self.canvas_height - self.court_height) // 2
        
        # データ読み込み
        self.load_data()
        
        # 再生制御変数
        self.current_frame = 0
        self.is_playing = False
        self.play_speed = 100  # ミリ秒
        
        # GUI要素の作成
        self.create_widgets()
        
        # 初期表示
        self.update_display()
    
    def load_data(self):
        """データの読み込み"""
        print("データを読み込んでいます...")
        
        # 選手座標データ
        self.player_data = pd.read_csv('data/transformed_player_points_offset.csv')
        print(f"選手データ: {len(self.player_data)} 行")
        
        # GKフェーズデータ
        self.phase_data = pd.read_csv('data/gk_phase_summary.csv')
        print(f"フェーズデータ: {len(self.phase_data)} フェーズ")
        
        # フレーム範囲
        self.min_frame = self.player_data['frame_num'].min()
        self.max_frame = self.player_data['frame_num'].max()
        self.current_frame = self.min_frame
        
        # ユニークなフレーム番号のリスト
        self.unique_frames = sorted(self.player_data['frame_num'].unique())
        
        # 現在のフェーズを特定
        self.current_phase_index = 0
        self.update_current_phase()
        
        # コート画像の読み込み
        self.load_court_images()
        
        print("データ読み込み完了")
    
    def load_court_images(self):
        """コート画像の読み込み"""
        try:
            # 右向きコート
            self.court_right_img = Image.open('data/court_right.png')
            self.court_right_img = self.court_right_img.resize((self.court_width, self.court_height))
            self.court_right_tk = ImageTk.PhotoImage(self.court_right_img)
            
            # 左向きコート
            self.court_left_img = Image.open('data/court_left.png')
            self.court_left_img = self.court_left_img.resize((self.court_width, self.court_height))
            self.court_left_tk = ImageTk.PhotoImage(self.court_left_img)
            
            print("コート画像読み込み完了")
        except Exception as e:
            print(f"コート画像読み込みエラー: {e}")
            # エラーの場合は白い背景を作成
            white_img = Image.new('RGB', (self.court_width, self.court_height), color='white')
            self.court_right_tk = ImageTk.PhotoImage(white_img)
            self.court_left_tk = ImageTk.PhotoImage(white_img)
    
    def create_widgets(self):
        """GUI要素の作成"""
        
        # メインフレーム
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 情報表示フレーム
        info_frame = ttk.Frame(main_frame)
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        # フレーム情報
        self.frame_label = ttk.Label(info_frame, text="フレーム: 0", font=("Arial", 12))
        self.frame_label.pack(side=tk.LEFT)
        
        # フェーズ情報
        self.phase_label = ttk.Label(info_frame, text="フェーズ: 0", font=("Arial", 12))
        self.phase_label.pack(side=tk.LEFT, padx=(20, 0))
        
        # ディレクション情報
        self.direction_label = ttk.Label(info_frame, text="方向: -", font=("Arial", 12))
        self.direction_label.pack(side=tk.LEFT, padx=(20, 0))
        
        # ディフェンスシステム情報
        self.defense_system_label = ttk.Label(info_frame, text="守備システム: -", font=("Arial", 12, "bold"))
        self.defense_system_label.pack(side=tk.LEFT, padx=(20, 0))
        
        # 9mライン外守備選手数
        self.outside_defenders_label = ttk.Label(info_frame, text="9m外: -人", font=("Arial", 12))
        self.outside_defenders_label.pack(side=tk.LEFT, padx=(20, 0))
        
        # キャンバスフレーム
        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        # キャンバス
        self.canvas = tk.Canvas(canvas_frame, width=self.canvas_width, height=self.canvas_height, bg='lightgray')
        self.canvas.pack()
        
        # コントロールフレーム
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(10, 0))
        
        # フェーズコントロール
        phase_frame = ttk.LabelFrame(control_frame, text="フェーズコントロール")
        phase_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Button(phase_frame, text="前フェーズ", command=self.prev_phase).pack(side=tk.LEFT, padx=2)
        ttk.Button(phase_frame, text="次フェーズ", command=self.next_phase).pack(side=tk.LEFT, padx=2)
        
        # フレームコントロール
        frame_control_frame = ttk.LabelFrame(control_frame, text="フレームコントロール")
        frame_control_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Button(frame_control_frame, text="<<", command=self.prev_frame).pack(side=tk.LEFT, padx=2)
        self.play_button = ttk.Button(frame_control_frame, text="再生", command=self.toggle_play)
        self.play_button.pack(side=tk.LEFT, padx=2)
        ttk.Button(frame_control_frame, text=">>", command=self.next_frame).pack(side=tk.LEFT, padx=2)
        
        # スピードコントロール
        speed_frame = ttk.LabelFrame(control_frame, text="再生速度")
        speed_frame.pack(fill=tk.X)
        
        ttk.Label(speed_frame, text="遅い").pack(side=tk.LEFT)
        self.speed_scale = ttk.Scale(speed_frame, from_=50, to=500, orient=tk.HORIZONTAL, 
                                   command=self.update_speed)
        self.speed_scale.set(self.play_speed)
        self.speed_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Label(speed_frame, text="速い").pack(side=tk.LEFT)
        
        # フレームスライダー
        slider_frame = ttk.LabelFrame(control_frame, text="フレーム位置")
        slider_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.frame_scale = ttk.Scale(slider_frame, from_=0, to=len(self.unique_frames)-1, 
                                   orient=tk.HORIZONTAL, command=self.on_frame_scale_change)
        self.frame_scale.pack(fill=tk.X, padx=5)
    
    def update_current_phase(self):
        """現在のフレームに対応するフェーズを特定"""
        for i, phase in self.phase_data.iterrows():
            if phase['start_frame'] <= self.current_frame <= phase['end_frame']:
                self.current_phase_index = i
                break
    
    def get_current_phase_info(self):
        """現在のフェーズ情報を取得"""
        if 0 <= self.current_phase_index < len(self.phase_data):
            return self.phase_data.iloc[self.current_phase_index]
        return None
    
    def convert_coordinates(self, x, y):
        """正規化座標(0-1)をキャンバス座標に変換"""
        # 0-1の範囲はコート画像内、それ以外は余白に表示
        canvas_x = x * self.court_width + self.court_offset_x
        canvas_y = y * self.court_height + self.court_offset_y
        return canvas_x, canvas_y
    
    def estimate_defense_system(self, frame_data, direction):
        """現在のフレームでディフェンスシステムを推定"""
        if direction is None:
            return None, 0
        
        def_color = defensive_team_color(direction)
        outside_count = 0
        
        for _, player in frame_data.iterrows():
            try:
                if str(player['team_color']).lower() != def_color:
                    continue
                
                x = float(player['x'])
                y = float(player['y'])
                
                if is_outside_9m(x, y, direction):
                    outside_count += 1
            except Exception:
                continue
        
        system = map_count_to_system(outside_count)
        return system, outside_count
    
    def draw_9m_line(self, direction):
        """9mラインを描画"""
        # y座標の範囲（0から1まで）
        y_values = np.linspace(0, 1, 100)
        points = []
        
        for y in y_values:
            if direction == 'left':
                boundary_x = boundary_x_for_left_defense(y)
            elif direction == 'right':
                boundary_x = boundary_x_for_right_defense(y)
            else:
                continue
            
            canvas_x, canvas_y = self.convert_coordinates(boundary_x, y)
            points.extend([canvas_x, canvas_y])
        
        if len(points) >= 4:  # 最低2点必要
            self.canvas.create_line(points, fill='red', width=2, tags='9m_line')
            
        # 9mライン説明
        self.canvas.create_text(self.canvas_width - 100, 60, 
                              text="赤線: 9mライン", 
                              font=("Arial", 10), fill='red', anchor='ne')
    
    def update_display(self, update_slider=True):
        """画面の更新"""
        # キャンバスをクリア
        self.canvas.delete("all")
        
        # 現在のフェーズ情報を取得
        phase_info = self.get_current_phase_info()
        current_direction = phase_info['direction'] if phase_info is not None else None
        
        # コート画像を描画（中央に配置）
        if phase_info is not None:
            direction = phase_info['direction']
            if direction == 'right':
                self.canvas.create_image(self.court_offset_x + self.court_width//2, 
                                       self.court_offset_y + self.court_height//2, 
                                       image=self.court_right_tk)
            else:
                self.canvas.create_image(self.court_offset_x + self.court_width//2, 
                                       self.court_offset_y + self.court_height//2, 
                                       image=self.court_left_tk)
        else:
            # フェーズ情報がない場合はright画像を使用
            self.canvas.create_image(self.court_offset_x + self.court_width//2, 
                                   self.court_offset_y + self.court_height//2, 
                                   image=self.court_right_tk)
        
        # コート範囲を示す枠線を描画（0-1の範囲）
        court_left = self.court_offset_x
        court_top = self.court_offset_y
        court_right = self.court_offset_x + self.court_width
        court_bottom = self.court_offset_y + self.court_height
        
        self.canvas.create_rectangle(court_left, court_top, court_right, court_bottom, 
                                   outline='blue', width=2, fill='')
        
        # 座標系の説明を表示
        self.canvas.create_text(self.canvas_width - 100, 20, 
                              text="青枠: 0-1範囲\n(コート内)", 
                              font=("Arial", 10), fill='blue', anchor='ne')
        
        # 9mラインを描画（現在の方向に基づく）
        if current_direction is not None:
            self.draw_9m_line(current_direction)
        
        # 現在のフレームの選手データを取得
        try:
            frame_data = self.player_data[self.player_data['frame_num'] == self.current_frame].copy()
        except Exception as e:
            print(f"データ取得エラー: {e}")
            frame_data = pd.DataFrame()
        
        # ディフェンスシステムを推定
        defense_system, outside_defenders_count = self.estimate_defense_system(frame_data, current_direction)
        
        # 選手をプロット
        for _, player in frame_data.iterrows():
            try:
                # 新しい座標変換を使用
                canvas_x, canvas_y = self.convert_coordinates(float(player['x']), float(player['y']))
                
                # 色を決定
                if int(player['is_gk']) == 1:
                    color = 'green'
                elif str(player['team_color']).lower() == 'red':
                    color = 'red'
                elif str(player['team_color']).lower() == 'white':
                    color = 'white'
                else:
                    color = 'blue'  # デフォルト色
                
                # 選手を描画（丸）
                radius = 8
                if color == 'white':
                    # 白い丸は黒い枠線付きで描画
                    self.canvas.create_oval(canvas_x-radius, canvas_y-radius, canvas_x+radius, canvas_y+radius, 
                                          fill=color, outline='black', width=2)
                else:
                    self.canvas.create_oval(canvas_x-radius, canvas_y-radius, canvas_x+radius, canvas_y+radius, 
                                          fill=color, outline='black', width=1)
                
                # 選手IDを表示
                self.canvas.create_text(canvas_x, canvas_y-radius-10, text=str(int(player['id'])), 
                                      font=("Arial", 8), fill='black')
            except Exception as e:
                print(f"選手描画エラー: {e}")
                continue
        
        # 情報ラベルを更新
        self.frame_label.config(text=f"フレーム: {self.current_frame}")
        
        if phase_info is not None:
            phase_num = self.current_phase_index + 1
            direction = phase_info['direction']
            start_frame = int(phase_info['start_frame'])
            end_frame = int(phase_info['end_frame'])
            self.phase_label.config(text=f"フェーズ: {phase_num} ({start_frame}-{end_frame})")
            self.direction_label.config(text=f"方向: {direction}")
            
            # ディフェンスシステム情報を更新
            if defense_system is not None:
                def_team = defensive_team_color(direction)
                self.defense_system_label.config(text=f"守備システム: {defense_system}")
                self.outside_defenders_label.config(text=f"9m外({def_team}): {outside_defenders_count}人")
            else:
                self.defense_system_label.config(text="守備システム: -")
                self.outside_defenders_label.config(text="9m外: -人")
        else:
            self.phase_label.config(text="フェーズ: -")
            self.direction_label.config(text="方向: -")
            self.defense_system_label.config(text="守備システム: -")
            self.outside_defenders_label.config(text="9m外: -人")
        
        # フレームスライダーを更新（無限ループを防ぐ）
        if update_slider:
            try:
                frame_index = self.unique_frames.index(self.current_frame)
                # 一時的にコールバックを無効化
                self.frame_scale.configure(command="")
                self.frame_scale.set(frame_index)
                self.frame_scale.configure(command=self.on_frame_scale_change)
            except (ValueError, tk.TclError):
                pass  # フレームがリストにない場合
    
    def prev_frame(self):
        """前のフレームに移動"""
        try:
            current_index = self.unique_frames.index(self.current_frame)
            if current_index > 0:
                self.current_frame = self.unique_frames[current_index - 1]
                self.update_current_phase()
                self.update_display()
        except ValueError:
            if self.unique_frames:
                self.current_frame = self.unique_frames[0]
                self.update_current_phase()
                self.update_display()
    
    def next_frame(self):
        """次のフレームに移動"""
        try:
            current_index = self.unique_frames.index(self.current_frame)
            if current_index < len(self.unique_frames) - 1:
                self.current_frame = self.unique_frames[current_index + 1]
                self.update_current_phase()
                self.update_display()
        except ValueError:
            if self.unique_frames:
                self.current_frame = self.unique_frames[0]
                self.update_current_phase()
                self.update_display()
    
    def prev_phase(self):
        """前のフェーズの開始フレームに移動"""
        if self.current_phase_index > 0:
            self.current_phase_index -= 1
            phase_info = self.phase_data.iloc[self.current_phase_index]
            self.current_frame = int(phase_info['start_frame'])
            self.update_display()
    
    def next_phase(self):
        """次のフェーズの開始フレームに移動"""
        if self.current_phase_index < len(self.phase_data) - 1:
            self.current_phase_index += 1
            phase_info = self.phase_data.iloc[self.current_phase_index]
            self.current_frame = int(phase_info['start_frame'])
            self.update_display()
    
    def toggle_play(self):
        """再生/停止を切り替え"""
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.play_button.config(text="停止")
            self.play_animation()
        else:
            self.play_button.config(text="再生")
    
    def play_animation(self):
        """アニメーション再生"""
        if self.is_playing:
            self.next_frame()
            # 最後のフレームに到達したら停止
            if self.current_frame >= self.max_frame:
                self.toggle_play()
            else:
                # 指定された間隔で次のフレームを再生
                self.root.after(self.play_speed, self.play_animation)
    
    def update_speed(self, value):
        """再生速度を更新"""
        self.play_speed = int(float(value))
    
    def on_frame_scale_change(self, value):
        """フレームスライダーの値が変更された時"""
        try:
            frame_index = int(float(value))
            if 0 <= frame_index < len(self.unique_frames):
                self.current_frame = self.unique_frames[frame_index]
                self.update_current_phase()
                self.update_display(update_slider=False)  # スライダー更新を無効化
        except (ValueError, IndexError) as e:
            print(f"フレームスライダーエラー: {e}")

def main():
    root = tk.Tk()
    app = HandballPlayerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
