import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

def draw_zone_polygons(ax, zones, zone_colors):
    """ゾーンポリゴンを描画"""
    for zone_num, polygon in zones.items():
        if polygon:
            # ポリゴンを描画
            poly = patches.Polygon(polygon, alpha=0.3, 
                                 facecolor=zone_colors.get(zone_num, 'gray'),
                                 edgecolor='black', linewidth=2)
            ax.add_patch(poly)
            
            # ゾーン番号をテキストで表示
            if len(polygon) > 0:
                center_x = np.mean([p[0] for p in polygon])
                center_y = np.mean([p[1] for p in polygon])
                ax.text(center_x, center_y, str(zone_num), 
                       fontsize=16, fontweight='bold', 
                       ha='center', va='center',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

def create_left_zones():
    """LEFT攻撃時のゾーン定義"""
    zones = {}
    
    # ゾーン1: x>=0.45, y>0.575
    zones[1] = [(0.45, 0.575), (1.0, 0.575), (1.0, 1.0), (0.45, 1.0)]
    
    # ゾーン2: x>=0.45, 0.425<=y<=0.575
    zones[2] = [(0.45, 0.425), (1.0, 0.425), (1.0, 0.575), (0.45, 0.575)]
    
    # ゾーン3: x>=0.45, y<0.425
    zones[3] = [(0.45, 0.0), (1.0, 0.0), (1.0, 0.425), (0.45, 0.425)]
    
    # ゾーン4: 拡張領域を含む
    zones[4] = [(-0.2, -0.2), (0.45, -0.2), (0.15, 0.275), (-0.2, 0.275)]
    
    # ゾーン5: 拡張領域を含む
    zones[5] = [(-0.2, 0.275), (0.15, 0.275), (0.45, 0.0), (0.45, 0.425), (-0.2, 0.425)]
    
    # ゾーン6: 拡張領域を含む
    zones[6] = [(-0.2, 0.425), (0.45, 0.425), (0.45, 0.575), (-0.2, 0.575)]
    
    # ゾーン7: 拡張領域を含む
    zones[7] = [(-0.2, 0.575), (0.45, 0.575), (0.45, 1.0), (0.15, 0.725), (-0.2, 0.725)]
    
    # ゾーン8: 拡張領域を含む
    zones[8] = [(-0.2, 0.725), (0.15, 0.725), (0.45, 1.2), (-0.2, 1.2)]
    
    return zones

def create_right_zones():
    """RIGHT攻撃時のゾーン定義（LEFT定義をx軸反転）"""
    left_zones = create_left_zones()
    right_zones = {}
    
    for zone_num, polygon in left_zones.items():
        # x座標を1-xで反転
        right_polygon = [(1 - x, y) for x, y in polygon]
        right_zones[zone_num] = right_polygon
    
    return right_zones

def visualize_zones():
    """ゾーンを可視化"""
    # カラーマップの定義
    zone_colors = {
        1: 'red',
        2: 'orange', 
        3: 'yellow',
        4: 'lightgreen',
        5: 'lightblue',
        6: 'blue',
        7: 'purple',
        8: 'pink'
    }
    
    # LEFT攻撃の可視化
    try:
        court_left = Image.open('data/court_left.png')
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # コート画像を表示
        ax.imshow(court_left, extent=[0, 1, 0, 1], aspect='auto', alpha=0.7)
        
        # LEFT攻撃のゾーンを描画
        left_zones = create_left_zones()
        draw_zone_polygons(ax, left_zones, zone_colors)
        
        ax.set_xlim(-0.3, 1.1)
        ax.set_ylim(-0.3, 1.3)
        ax.set_xlabel('X Coordinate', fontsize=12)
        ax.set_ylabel('Y Coordinate', fontsize=12)
        ax.set_title('LEFT Attack Zone Definition', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 凡例を追加
        legend_elements = [patches.Patch(facecolor=zone_colors[i], label=f'Zone {i}') 
                          for i in range(1, 9)]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))
        
        plt.tight_layout()
        plt.savefig('data/zones_left_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    except FileNotFoundError:
        print("Warning: data/court_left.png not found. Creating zones without background image.")
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        left_zones = create_left_zones()
        draw_zone_polygons(ax, left_zones, zone_colors)
        ax.set_xlim(-0.3, 1.1)
        ax.set_ylim(-0.3, 1.3)
        ax.set_xlabel('X Coordinate', fontsize=12)
        ax.set_ylabel('Y Coordinate', fontsize=12)
        ax.set_title('LEFT Attack Zone Definition', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        legend_elements = [patches.Patch(facecolor=zone_colors[i], label=f'Zone {i}') 
                          for i in range(1, 9)]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))
        plt.tight_layout()
        plt.savefig('data/zones_left_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # RIGHT攻撃の可視化
    try:
        court_right = Image.open('data/court_right.png')
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # コート画像を表示
        ax.imshow(court_right, extent=[0, 1, 0, 1], aspect='auto', alpha=0.7)
        
        # RIGHT攻撃のゾーンを描画
        right_zones = create_right_zones()
        draw_zone_polygons(ax, right_zones, zone_colors)
        
        ax.set_xlim(-0.1, 1.3)
        ax.set_ylim(-0.3, 1.3)
        ax.set_xlabel('X Coordinate', fontsize=12)
        ax.set_ylabel('Y Coordinate', fontsize=12)
        ax.set_title('RIGHT Attack Zone Definition', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 凡例を追加
        legend_elements = [patches.Patch(facecolor=zone_colors[i], label=f'Zone {i}') 
                          for i in range(1, 9)]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(-0.02, 1))
        
        plt.tight_layout()
        plt.savefig('data/zones_right_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    except FileNotFoundError:
        print("Warning: data/court_right.png not found. Creating zones without background image.")
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        right_zones = create_right_zones()
        draw_zone_polygons(ax, right_zones, zone_colors)
        ax.set_xlim(-0.1, 1.3)
        ax.set_ylim(-0.3, 1.3)
        ax.set_xlabel('X Coordinate', fontsize=12)
        ax.set_ylabel('Y Coordinate', fontsize=12)
        ax.set_title('RIGHT Attack Zone Definition', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        legend_elements = [patches.Patch(facecolor=zone_colors[i], label=f'Zone {i}') 
                          for i in range(1, 9)]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(-0.02, 1))
        plt.tight_layout()
        plt.savefig('data/zones_right_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()

def print_zone_info():
    """ゾーン定義の詳細情報を出力"""
    print("=== LEFT攻撃時のゾーン定義 ===")
    left_zones = create_left_zones()
    for zone_num, polygon in left_zones.items():
        print(f"ゾーン{zone_num}: {polygon}")
    
    print("\n=== RIGHT攻撃時のゾーン定義 ===")
    right_zones = create_right_zones()
    for zone_num, polygon in right_zones.items():
        print(f"ゾーン{zone_num}: {polygon}")

if __name__ == "__main__":
    print("ハンドボール攻撃ゾーンの可視化を開始...")
    print_zone_info()
    print("\nゾーンの可視化を実行中...")
    visualize_zones()
    print("可視化完了！")
    print("保存ファイル:")
    print("- data/zones_left_visualization.png")
    print("- data/zones_right_visualization.png")
