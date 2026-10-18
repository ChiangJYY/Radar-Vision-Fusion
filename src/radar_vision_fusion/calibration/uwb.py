"""UWB anchor calibration from caller-supplied distances."""
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class UWBCalibration:
    def __init__(self):
        """UWB基站标定类"""
        self.anchor_positions = {}  # 存储基站坐标

    def set_anchor_distances(self, distances):
        """
        设置各基站之间的距离测量值

        参数:
        distances: dict, 包含各基站间距离的字典
                  例如: {'a0_a1': 1.5, 'a0_a2': 2.1, 'a0_a3': 2.8,
                        'a1_a2': 1.8, 'a1_a3': 2.5, 'a2_a3': 1.2}
        """
        self.distances = distances

    def calibrate_anchors(self):
        """
        基于距离测量值计算各基站坐标

        标定策略:
        - a0: (0, 0, 0) - 坐标原点
        - a1: (d01, 0, 0) - x轴正方向
        - a2: 通过几何计算确定位置
        - a3: 通过几何计算确定位置
        """
        # 检查必要的距离测量值是否存在
        required_distances = ['a0_a1', 'a0_a2', 'a0_a3', 'a1_a2', 'a1_a3', 'a2_a3']
        for dist_key in required_distances:
            if dist_key not in self.distances:
                raise ValueError(f"缺少距离测量值: {dist_key}")

        # 获取距离值
        d01 = self.distances['a0_a1']
        d02 = self.distances['a0_a2']
        d03 = self.distances['a0_a3']
        d12 = self.distances['a1_a2']
        d13 = self.distances['a1_a3']
        d23 = self.distances['a2_a3']

        # 设置a0和a1的位置
        self.anchor_positions['a0'] = np.array([0.0, 0.0, 0.0])
        self.anchor_positions['a1'] = np.array([d01, 0.0, 0.0])

        # 计算a2的位置
        # a2在xOy平面内，通过a0和a1的距离约束确定
        a2_pos = self._calculate_anchor2_position(d01, d02, d12)
        self.anchor_positions['a2'] = a2_pos

        # 计算a3的位置
        # a3在三维空间中，通过与其他三个点的距离约束确定
        a3_pos = self._calculate_anchor3_position(d01, d02, d03, d13, d23)
        self.anchor_positions['a3'] = a3_pos

        return self.anchor_positions

    def _calculate_anchor2_position(self, d01, d02, d12):
        """
        计算a2的位置（在xOy平面内）

        使用几何方法：通过a0、a1、a2形成三角形
        a0在(0,0,0)，a1在(d01,0,0)，a2在(x,y,0)
        """
        # 使用余弦定理计算a2的x坐标
        # cos(angle_at_a0) = (d01^2 + d02^2 - d12^2) / (2 * d01 * d02)
        cos_angle_a0 = (d01**2 + d02**2 - d12**2) / (2 * d01 * d02)

        # 确保cos值在有效范围内
        cos_angle_a0 = np.clip(cos_angle_a0, -1, 1)

        x2 = d02 * cos_angle_a0
        y2 = d02 * math.sqrt(1 - cos_angle_a0**2)

        return np.array([x2, y2, 0.0])

    def _calculate_anchor3_position(self, d01, d02, d03, d13, d23):
        """
        计算a3的位置（三维空间）

        使用最小二乘法或几何方法求解三维点位置
        """
        # 方法1：使用空间几何计算
        # 我们有三个参考点a0, a1, a2，需要找到与它们距离分别为d03, d13, d23的点a3

        p0 = self.anchor_positions['a0']
        p1 = self.anchor_positions['a1']
        p2 = self.anchor_positions['a2']

        # 使用三球定位法（trilateration）
        # 解方程组：
        # ||a3 - p0|| = d03
        # ||a3 - p1|| = d13
        # ||a3 - p2|| = d23

        a3_pos = self._trilaterate_3d(p0, p1, p2, d03, d13, d23)
        return a3_pos

    def _trilaterate_3d(self, p0, p1, p2, d0, d1, d2):
        """
        三维三球定位算法

        参数:
        p0, p1, p2: 参考点坐标
        d0, d1, d2: 到各参考点的距离
        """
        # 将坐标原点移到p1
        ex = (p0 - p1) / np.linalg.norm(p0 - p1)
        i = np.dot(ex, p2 - p1)
        ey = (p2 - p1 - i*ex) / np.linalg.norm(p2 - p1 - i*ex)
        ez = np.cross(ex, ey)
        d = np.linalg.norm(p0 - p1)
        j = np.dot(ey, p2 - p1)

        # 解二次方程
        x = (d0**2 - d1**2 + d**2) / (2*d)
        y = (d0**2 - d2**2 + i**2 + j**2) / (2*j) - (i/j)*x
        z_squared = d0**2 - x**2 - y**2

        if z_squared < 0:
            # 处理数值误差
            z_squared = 0

        z = math.sqrt(z_squared)

        # 转换回原始坐标系
        result = p1 + x*ex + y*ey + z*ez
        return result

    def get_anchor_positions(self):
        """获取所有基站坐标"""
        return self.anchor_positions

    def save_calibration_results(self, filename='uwb_calibration_results.txt'):
        """保存标定结果到文件"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("UWB基站标定结果\n")
            f.write("=" * 30 + "\n")
            f.write("坐标系说明：\n")
            f.write("- a0为坐标原点 (0,0,0)\n")
            f.write("- a0到a1为x轴正方向\n")
            f.write("- 使用激光测距测量基站间距离\n")
            f.write("\n")
            f.write("距离测量值：\n")
            for key, value in self.distances.items():
                f.write(f"{key}: {value:.3f} m\n")
            f.write("\n")
            f.write("基站坐标：\n")
            for anchor, pos in self.anchor_positions.items():
                f.write(f"{anchor}: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}) m\n")

        print(f"标定结果已保存到 {filename}")

    def input_distances_interactive(self):
        """
        交互式输入距离测量值

        返回:
        distances: dict, 包含所有基站间距离的字典
        """
        print("\n=== UWB基站距离测量输入 ===")
        print("请使用激光测距仪测量以下距离，并输入数值（单位：米）")
        print("注意：确保测量精度在0.01米以内\n")

        distances = {}
        anchor_pairs = [
            ('a0', 'a1'),
            ('a0', 'a2'),
            ('a0', 'a3'),
            ('a1', 'a2'),
            ('a1', 'a3'),
            ('a2', 'a3')
        ]

        for anchor_i, anchor_j in anchor_pairs:
            while True:
                try:
                    key = f"{anchor_i}_{anchor_j}"
                    prompt = f"请输入 {anchor_i} 到 {anchor_j} 的距离 (米): "
                    distance = float(input(prompt))

                    if distance <= 0:
                        print("错误：距离必须大于0，请重新输入")
                        continue

                    # 基本几何验证（三角不等式）
                    if self._validate_distance_triangle(anchor_i, anchor_j, distance, distances):
                        distances[key] = distance
                        print(".3f")
                        break
                    else:
                        print("警告：该距离可能与已输入的距离不一致，可能导致标定误差")
                        confirm = input("是否仍要使用该值？(y/N): ").lower().strip()
                        if confirm == 'y':
                            distances[key] = distance
                            print(".3f")
                            break

                except ValueError:
                    print("错误：请输入有效的数字")

        print("\n所有距离输入完成！")
        print("输入的距离值：")
        for key, value in distances.items():
            print(".3f")

        return distances

    def _validate_distance_triangle(self, anchor_i, anchor_j, distance, current_distances):
        """
        验证距离是否满足三角不等式
        """
        # 这里可以添加更复杂的几何验证逻辑
        # 暂时只做基本的正值检查
        return distance > 0

    def visualize_calibration_2d(self, show_distances=True, show_labels=True):
        """
        2D可视化UWB基站标定结果（XY平面俯视图）

        参数:
        show_distances: bool, 是否显示距离标签
        show_labels: bool, 是否显示基站标签
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        # 设置颜色和标记
        colors = ['red', 'blue', 'green', 'orange']
        markers = ['o', 's', '^', 'D']

        anchor_names = ['a0', 'a1', 'a2', 'a3']

        # 获取所有坐标的最大范围，用于设置轴的限制
        all_x = [pos[0] for pos in self.anchor_positions.values()]
        all_y = [pos[1] for pos in self.anchor_positions.values()]

        margin = 0.5
        x_range = [min(all_x) - margin, max(all_x) + margin]
        y_range = [min(all_y) - margin, max(all_y) + margin]

        # 绘制XY平面俯视图
        for i, (anchor, pos) in enumerate(self.anchor_positions.items()):
            ax.scatter(pos[0], pos[1], c=colors[i], marker=markers[i], s=150, alpha=0.8,
                       edgecolors='black', linewidth=2, label=f'基站 {anchor}')
            if show_labels:
                ax.annotate(f'{anchor}\n({pos[0]:.2f}, {pos[1]:.2f})',
                           (pos[0], pos[1]), xytext=(5, 5), textcoords='offset points',
                           fontsize=10, ha='left', bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

        # 绘制区域边界线（四边形区域）
        # 定义区域边界顺序：a0 -> a1 -> a2 -> a3 -> a0
        region_boundary = ['a0', 'a1', 'a2', 'a3', 'a0']

        # 绘制边界线
        for i in range(len(region_boundary) - 1):
            anchor_i = region_boundary[i]
            anchor_j = region_boundary[i + 1]

            if anchor_i in self.anchor_positions and anchor_j in self.anchor_positions:
                pos_i = self.anchor_positions[anchor_i]
                pos_j = self.anchor_positions[anchor_j]

                # 绘制边界线
                ax.plot([pos_i[0], pos_j[0]], [pos_i[1], pos_j[1]], 'blue', alpha=0.8, linewidth=3, solid_capstyle='round')

                # 显示距离标签
                if show_distances:
                    mid_x, mid_y = (pos_i[0] + pos_j[0]) / 2, (pos_i[1] + pos_j[1]) / 2
                    dist_key = f"{anchor_i}_{anchor_j}"
                    if dist_key in self.distances:
                        distance = self.distances[dist_key]
                        ax.annotate('.2f', (mid_x, mid_y),
                                  xytext=(0, 5), textcoords='offset points',
                                  fontsize=9, ha='center', va='bottom',
                                  bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
                                  fontweight='bold')

        # 设置坐标轴标签和标题
        ax.set_xlabel('X (米)', fontsize=12)
        ax.set_ylabel('Y (米)', fontsize=12)
        ax.set_title('UWB基站标定结果 - XY平面俯视图\n(a0为原点，a0→a1为X轴正方向)',
                    fontsize=14, fontweight='bold', pad=20)

        # 设置轴的限制和等比例缩放
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')
        ax.set_aspect('equal')

        # 添加坐标系说明
        ax.text(0.02, 0.98, '坐标系说明:', transform=ax.transAxes, fontsize=10, fontweight='bold',
                verticalalignment='top')
        ax.text(0.02, 0.94, '• a0: 坐标原点 (0,0)', transform=ax.transAxes, fontsize=9)
        ax.text(0.02, 0.90, '• a0→a1: X轴正方向', transform=ax.transAxes, fontsize=9)
        ax.text(0.02, 0.86, '• 距离单位: 米', transform=ax.transAxes, fontsize=9)

        plt.tight_layout()
        plt.show()


    def validate_calibration(self, tolerance=0.01):
        """
        验证标定结果的正确性

        检查计算出的距离是否与测量距离一致
        """
        print("标定验证结果：")
        print("-" * 30)

        all_valid = True
        for i, anchor_i in enumerate(['a0', 'a1', 'a2', 'a3']):
            for j, anchor_j in enumerate(['a0', 'a1', 'a2', 'a3']):
                if i < j:
                    dist_key = f"{anchor_i}_{anchor_j}"
                    if dist_key in self.distances:
                        measured_dist = self.distances[dist_key]
                        calculated_dist = np.linalg.norm(
                            self.anchor_positions[anchor_i] - self.anchor_positions[anchor_j]
                        )

                        error = abs(calculated_dist - measured_dist)
                        is_valid = error < tolerance

                        status = "OK" if is_valid else "ERROR"
                        print(f"{dist_key}: 测量={measured_dist:.3f}, 计算={calculated_dist:.3f}, 误差={error:.3f} {status}")
                        if not is_valid:
                            all_valid = False

        return all_valid
