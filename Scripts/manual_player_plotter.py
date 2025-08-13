"""
ハンドボールの選手の位置を手動でプロットするためのGUIアプリケーション。
python Scripts/manual_player_plotter.py
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
from PIL import Image, ImageTk
import os

class PlayerPlotterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Handball Player Position Plotter")
        self.root.geometry("1200x800")
        
        # データ保存用
        self.saved_data = []  # 保存されたデータ
        self.current_frame = 1
        
        # 現在のフレームでの選手配置（常に6人ずつ）
        self.current_players = {
            'defense': [],  # [(id, x, y, canvas_id), ...]
            'attack': []
        }
        
        # 画像関連
        self.court_image = None
        self.photo = None
        self.canvas_width = 800
        self.canvas_height = 600
        
        # プロット設定
        self.current_direction = "left"  # left or right
        self.selected_team = 'defense'  # 選択されたチーム
        
        # ドラッグ関連
        self.dragging_item = None
        self.drag_start_x = 0
        self.drag_start_y = 0
        
        self.setup_ui()
        self.load_court_image()
        self.initialize_players()  # 初期配置
        
    def setup_ui(self):
        """UIを設定"""
        # メインフレーム
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左側：コントロールパネル
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # 右側：キャンバス
        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # コントロールパネル
        self.setup_controls(control_frame)
        
        # キャンバス
        self.setup_canvas(canvas_frame)
        
    def setup_controls(self, parent):
        """コントロールパネルを設定"""
        # タイトル
        title_label = ttk.Label(parent, text="Player Plotter Controls", font=("Arial", 14, "bold"))
        title_label.pack(pady=(0, 20))
        
        # フレーム番号
        frame_frame = ttk.LabelFrame(parent, text="Frame Settings")
        frame_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(frame_frame, text="Frame Number:").pack(anchor=tk.W)
        self.frame_var = tk.IntVar(value=1)
        frame_spinbox = ttk.Spinbox(frame_frame, from_=1, to=999999, textvariable=self.frame_var, width=10)
        frame_spinbox.pack(pady=(0, 5))
        
        # フレーム変更ボタン
        ttk.Button(frame_frame, text="Load Frame", command=self.load_frame).pack(pady=(5, 0))
        
        # 方向選択
        direction_frame = ttk.LabelFrame(parent, text="Court Direction")
        direction_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.direction_var = tk.StringVar(value="left")
        ttk.Radiobutton(direction_frame, text="Left Attack", variable=self.direction_var, 
                       value="left", command=self.on_direction_change).pack(anchor=tk.W)
        ttk.Radiobutton(direction_frame, text="Right Attack", variable=self.direction_var, 
                       value="right", command=self.on_direction_change).pack(anchor=tk.W)
        
        # チーム選択
        team_frame = ttk.LabelFrame(parent, text="Player Type")
        team_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.team_var = tk.StringVar(value="defense")
        ttk.Radiobutton(team_frame, text="Defense (Blue)", variable=self.team_var, 
                       value="defense").pack(anchor=tk.W)
        ttk.Radiobutton(team_frame, text="Attack (Red)", variable=self.team_var, 
                       value="attack").pack(anchor=tk.W)
        
        # 選手数表示
        self.player_count_label = ttk.Label(team_frame, text="Defense: 6/6, Attack: 6/6")
        self.player_count_label.pack(pady=(5, 0))
        
        # 座標表示
        coord_frame = ttk.LabelFrame(parent, text="Current Coordinates")
        coord_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.coord_label = ttk.Label(coord_frame, text="X: -, Y: -")
        self.coord_label.pack(pady=5)
        
        # 操作説明
        help_frame = ttk.LabelFrame(parent, text="Controls")
        help_frame.pack(fill=tk.X, pady=(0, 10))
        
        help_text = "• Left click: Add player\n• Drag: Move player\n• Right click: Delete player"
        ttk.Label(help_frame, text=help_text, font=("Arial", 8)).pack(pady=5)
        
        # ボタン
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(button_frame, text="Initialize Players", command=self.initialize_players).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(button_frame, text="Clear Current Frame", command=self.clear_current_frame).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(button_frame, text="Save Current Frame", command=self.save_current_frame).pack(fill=tk.X, pady=(0, 5))
        
        # データ管理
        data_frame = ttk.LabelFrame(parent, text="Data Management")
        data_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(data_frame, text="Load Data", command=self.load_data).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(data_frame, text="Export All CSV", command=self.export_csv).pack(fill=tk.X, pady=(0, 5))
        
        # データ表示
        data_display_frame = ttk.LabelFrame(parent, text="Saved Data")
        data_display_frame.pack(fill=tk.BOTH, expand=True)
        
        # データ一覧表示用のTreeview
        columns = ("Frame", "ID", "Team", "Direction", "X", "Y")
        self.data_tree = ttk.Treeview(data_display_frame, columns=columns, show="headings", height=8)
        
        for col in columns:
            self.data_tree.heading(col, text=col)
            self.data_tree.column(col, width=60)
        
        scrollbar = ttk.Scrollbar(data_display_frame, orient=tk.VERTICAL, command=self.data_tree.yview)
        self.data_tree.configure(yscrollcommand=scrollbar.set)
        
        self.data_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
    def setup_canvas(self, parent):
        """キャンバスを設定"""
        canvas_label = ttk.Label(parent, text="Court View", font=("Arial", 14, "bold"))
        canvas_label.pack()
        
        # キャンバス作成
        self.canvas = tk.Canvas(parent, width=self.canvas_width, height=self.canvas_height, 
                               bg="white", relief=tk.SUNKEN, borderwidth=2)
        self.canvas.pack(pady=10)
        
        # マウスイベント
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Button-3>", self.on_right_click)  # 右クリックで削除
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        
    def load_court_image(self):
        """コート画像を読み込み"""
        direction = self.direction_var.get()
        image_path = f"data/court_{direction}.png"
        
        try:
            # 画像を読み込み、キャンバスサイズにリサイズ
            image = Image.open(image_path)
            image = image.resize((self.canvas_width, self.canvas_height), Image.Resampling.LANCZOS)
            self.photo = ImageTk.PhotoImage(image)
            
            # キャンバスに画像を表示
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
            
            # 既存のプロットを再描画
            self.redraw_current_players()
            
        except FileNotFoundError:
            messagebox.showwarning("Warning", f"Court image not found: {image_path}")
            self.canvas.delete("all")
            self.canvas.create_text(self.canvas_width//2, self.canvas_height//2, 
                                  text=f"Court image not found\n{image_path}", 
                                  font=("Arial", 12), fill="red")
    
    def initialize_players(self):
        """初期選手配置（6人ずつ）"""
        # 既存の選手をクリア
        self.current_players = {'defense': [], 'attack': []}
        self.canvas.delete("player_point")
        
        # 守備選手の初期配置（青）
        defense_positions = [
            (0.2, 0.2), (0.2, 0.4), (0.2, 0.6), (0.2, 0.8),
            (0.1, 0.3), (0.1, 0.7)
        ]
        
        # 攻撃選手の初期配置（赤）
        attack_positions = [
            (0.7, 0.2), (0.7, 0.4), (0.7, 0.6), (0.7, 0.8),
            (0.8, 0.3), (0.8, 0.7)
        ]
        
        # 守備選手を配置
        for i, (x_norm, y_norm) in enumerate(defense_positions):
            x = x_norm * self.canvas_width
            y = y_norm * self.canvas_height
            canvas_id = self.draw_point(x, y, "defense", i + 1)
            self.current_players['defense'].append((i + 1, x_norm, y_norm, canvas_id))
        
        # 攻撃選手を配置
        for i, (x_norm, y_norm) in enumerate(attack_positions):
            x = x_norm * self.canvas_width
            y = y_norm * self.canvas_height
            canvas_id = self.draw_point(x, y, "attack", i + 1)
            self.current_players['attack'].append((i + 1, x_norm, y_norm, canvas_id))
        
        self.update_player_count()
    
    def on_direction_change(self):
        """方向変更時の処理"""
        self.load_court_image()
    
    def on_canvas_click(self, event):
        """キャンバスクリック時の処理"""
        # 既存の点をクリックした場合はドラッグ開始
        clicked_item = self.canvas.find_closest(event.x, event.y)[0]
        if "player_circle" in self.canvas.gettags(clicked_item):
            self.dragging_item = clicked_item
            self.drag_start_x = event.x
            self.drag_start_y = event.y
            return
        
        # 新しい点を配置
        team = self.team_var.get()
        
        # 選手数制限チェック
        if len(self.current_players[team]) >= 6:
            messagebox.showwarning("Warning", f"Maximum 6 {team} players allowed!")
            return
        
        # キャンバス座標を0-1の範囲に正規化
        x_norm = event.x / self.canvas_width
        y_norm = event.y / self.canvas_height
        
        # 範囲チェック
        if not (0 <= x_norm <= 1 and 0 <= y_norm <= 1):
            return
        
        # 新しい選手IDを生成（既存IDの最大値+1）
        existing_ids = [pid for pid, _, _, _ in self.current_players[team]]
        if existing_ids:
            player_id = max(existing_ids) + 1
        else:
            player_id = 1
        
        # プロットを描画
        canvas_id = self.draw_point(event.x, event.y, team, player_id)
        
        # 現在のフレームデータに追加
        self.current_players[team].append((player_id, x_norm, y_norm, canvas_id))
        
        # 選手数表示を更新
        self.update_player_count()
    
    def draw_point(self, x, y, team, player_id):
        """点を描画"""
        color = "blue" if team == "defense" else "red"
        radius = 8  # 点を大きくした
        
        circle_id = self.canvas.create_oval(x-radius, y-radius, x+radius, y+radius, 
                                          fill=color, outline="black", width=2, 
                                          tags=("player_point", "player_circle"))
        
        # IDを表示
        self.canvas.create_text(x, y-20, text=str(player_id), 
                               font=("Arial", 10, "bold"), fill="black", 
                               tags=("player_point", "player_text"))
        
        return circle_id
    
    def on_right_click(self, event):
        """右クリックで選手を削除"""
        clicked_item = self.canvas.find_closest(event.x, event.y)[0]
        if "player_circle" in self.canvas.gettags(clicked_item):
            # 対応する選手を削除
            for team in ['defense', 'attack']:
                for i, (player_id, x, y, canvas_id) in enumerate(self.current_players[team]):
                    if canvas_id == clicked_item:
                        # 選手データから削除
                        self.current_players[team].pop(i)
                        
                        # キャンバスから削除（円とテキスト両方）
                        self.canvas.delete(canvas_id)
                        
                        # 関連するテキストも削除
                        for item in self.canvas.find_all():
                            if "player_text" in self.canvas.gettags(item):
                                text_coords = self.canvas.coords(item)
                                if len(text_coords) >= 2:
                                    # 削除された円の近くのテキストを探して削除
                                    text_x, text_y = text_coords[0], text_coords[1]
                                    if abs(text_x - event.x) < 20 and abs(text_y - event.y + 20) < 20:
                                        self.canvas.delete(item)
                                        break
                        
                        self.update_player_count()
                        return
    
    def on_drag(self, event):
        """ドラッグ中の処理"""
        if self.dragging_item:
            # 移動量を計算
            dx = event.x - self.drag_start_x
            dy = event.y - self.drag_start_y
            
            # 点を移動
            self.canvas.move(self.dragging_item, dx, dy)
            
            # 関連するテキストも移動
            for item in self.canvas.find_all():
                if "player_text" in self.canvas.gettags(item):
                    # テキストの位置をチェックして、対応する円の近くにあるかどうか確認
                    text_coords = self.canvas.coords(item)
                    circle_coords = self.canvas.coords(self.dragging_item)
                    if len(text_coords) >= 2 and len(circle_coords) >= 4:
                        text_x, text_y = text_coords[0], text_coords[1]
                        circle_center_x = (circle_coords[0] + circle_coords[2]) / 2
                        circle_center_y = (circle_coords[1] + circle_coords[3]) / 2
                        # テキストが円の近くにある場合は一緒に移動
                        if abs(text_x - circle_center_x) < 20 and abs(text_y - circle_center_y + 20) < 20:
                            self.canvas.move(item, dx, dy)
                            break
            
            # 新しいドラッグ開始点を設定
            self.drag_start_x = event.x
            self.drag_start_y = event.y
    
    def on_release(self, event):
        """ドラッグ終了時の処理"""
        if self.dragging_item:
            # 新しい座標を正規化して保存
            coords = self.canvas.coords(self.dragging_item)
            if len(coords) >= 4:
                center_x = (coords[0] + coords[2]) / 2
                center_y = (coords[1] + coords[3]) / 2
                
                x_norm = center_x / self.canvas_width
                y_norm = center_y / self.canvas_height
                
                # 範囲内に制限
                x_norm = max(0, min(1, x_norm))
                y_norm = max(0, min(1, y_norm))
                
                # 対応するプレイヤーデータを更新
                self.update_player_position(self.dragging_item, x_norm, y_norm)
            
            self.dragging_item = None
    
    def update_player_position(self, canvas_id, x_norm, y_norm):
        """選手位置を更新"""
        for team in ['defense', 'attack']:
            for i, (player_id, x, y, c_id) in enumerate(self.current_players[team]):
                if c_id == canvas_id:
                    self.current_players[team][i] = (player_id, x_norm, y_norm, c_id)
                    return
    
    def on_mouse_move(self, event):
        """マウス移動時の座標表示"""
        x_norm = round(event.x / self.canvas_width, 4)
        y_norm = round(event.y / self.canvas_height, 4)
        self.coord_label.config(text=f"X: {x_norm}, Y: {y_norm}")
    
    def redraw_current_players(self):
        """現在のフレームの選手を再描画"""
        # プロット関連のタグを削除
        self.canvas.delete("player_point")
        
        # 現在の方向と一致する選手のみ再描画
        current_direction = self.direction_var.get()
        for team in ['defense', 'attack']:
            for player_id, x_norm, y_norm, _ in self.current_players[team]:
                x = x_norm * self.canvas_width
                y = y_norm * self.canvas_height
                canvas_id = self.draw_point(x, y, team, player_id)
                
                # canvas_idを更新
                for i, (pid, xn, yn, _) in enumerate(self.current_players[team]):
                    if pid == player_id:
                        self.current_players[team][i] = (pid, xn, yn, canvas_id)
                        break
        
        self.update_player_count()
    
    def update_player_count(self):
        """選手数表示を更新（常に6/6）"""
        defense_count = len(self.current_players['defense'])
        attack_count = len(self.current_players['attack'])
        self.player_count_label.config(text=f"Defense: {defense_count}/6, Attack: {attack_count}/6")
    
    def reset_positions(self):
        """選手位置をリセット"""
        if messagebox.askyesno("Confirm", "Reset all player positions to default?"):
            self.current_players = {'defense': [], 'attack': []}
            self.canvas.delete("player_point")
            self.initialize_players()
    
    def load_frame(self):
        """指定フレームのデータを読み込み"""
        frame_num = self.frame_var.get()
        direction = self.direction_var.get()
        
        # 現在の選手をクリア
        self.current_players = {'defense': [], 'attack': []}
        self.canvas.delete("player_point")
        
        # 保存されたデータから該当フレームを検索
        frame_data = {'defense': {}, 'attack': {}}
        for data in self.saved_data:
            if data['frame_num'] == frame_num and data['direction'] == direction:
                team = data['team']
                player_id = data['player_id']
                frame_data[team][player_id] = (data['x'], data['y'])
        
        # 各チーム6人ずつ配置
        for team in ['defense', 'attack']:
            for player_id in range(1, 7):  # 1-6のID
                if player_id in frame_data[team]:
                    # 保存されたデータから位置を取得
                    x_norm, y_norm = frame_data[team][player_id]
                else:
                    # デフォルト位置を使用
                    if team == 'defense':
                        default_positions = [(0.2, 0.2), (0.2, 0.4), (0.2, 0.6), (0.2, 0.8), (0.1, 0.3), (0.1, 0.7)]
                    else:
                        default_positions = [(0.7, 0.2), (0.7, 0.4), (0.7, 0.6), (0.7, 0.8), (0.8, 0.3), (0.8, 0.7)]
                    x_norm, y_norm = default_positions[player_id - 1]
                
                # 選手を配置
                x = x_norm * self.canvas_width
                y = y_norm * self.canvas_height
                canvas_id = self.draw_point(x, y, team, player_id)
                self.current_players[team].append((player_id, x_norm, y_norm, canvas_id))
        
        self.update_player_count()
    
    def clear_current_frame(self):
        """現在のフレームの選手を全てクリア"""
        self.current_players = {'defense': [], 'attack': []}
        # キャンバスから選手関連の要素を削除
        for item in self.canvas.find_all():
            if "player_circle" in self.canvas.gettags(item) or "player_text" in self.canvas.gettags(item):
                self.canvas.delete(item)
        self.update_player_count()
    
    def save_current_frame(self):
        """現在のフレームを保存"""
        frame_num = self.frame_var.get()
        direction = self.direction_var.get()
        
        # 既存の同フレーム・同方向データを削除
        self.saved_data = [d for d in self.saved_data 
                          if not (d['frame_num'] == frame_num and d['direction'] == direction)]
        
        # 現在の選手データを保存
        for team in ['defense', 'attack']:
            for player_id, x_norm, y_norm, _ in self.current_players[team]:
                data_point = {
                    'frame_num': frame_num,
                    'player_id': player_id,
                    'team': team,
                    'direction': direction,
                    'x': round(x_norm, 4),
                    'y': round(y_norm, 4)
                }
                self.saved_data.append(data_point)
        
        self.update_data_display()
        messagebox.showinfo("Success", f"Frame {frame_num} saved!")
    
    def update_data_display(self):
        """データ表示を更新"""
        # 既存のアイテムをクリア
        for item in self.data_tree.get_children():
            self.data_tree.delete(item)
        
        # データを追加
        for i, data in enumerate(sorted(self.saved_data, key=lambda x: (x['frame_num'], x['team'], x['player_id']))):
            self.data_tree.insert("", "end", values=(
                data['frame_num'],
                data['player_id'],
                data['team'],
                data['direction'],
                data['x'],
                data['y']
            ))
    
    def load_data(self):
        """データを読み込み"""
        filename = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Load plot data"
        )
        
        if filename:
            try:
                df = pd.read_csv(filename)
                self.saved_data = df.to_dict('records')
                self.update_data_display()
                messagebox.showinfo("Success", f"Data loaded from {filename}")
                
                # 現在のフレームを再読み込み
                self.load_frame()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load data: {str(e)}")
    
    def export_csv(self):
        """CSVにエクスポート"""
        if not self.saved_data:
            messagebox.showwarning("Warning", "No data to export!")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            title="Export to CSV",
            initialfile="manual_player_positions.csv"
        )
        
        if filename:
            try:
                df = pd.DataFrame(self.saved_data)
                # 列順を整理
                df = df[['frame_num', 'player_id', 'team', 'direction', 'x', 'y']]
                df = df.sort_values(['frame_num', 'team', 'player_id'])
                df.to_csv(filename, index=False)
                messagebox.showinfo("Success", f"Data exported to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export data: {str(e)}")

def main():
    root = tk.Tk()
    app = PlayerPlotterGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
