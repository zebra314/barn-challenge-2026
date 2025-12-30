import numpy as np
import math

class LidarCluster:
    def __init__(self, cluster_threshold=0.5, min_points=3):
        """
        cluster_threshold: 兩點距離小於此值視為同一物體 (公尺)
        min_points: 一個群集至少要有多少點才算有效 (過濾雜訊)
        """
        self.threshold = cluster_threshold
        self.min_points = min_points

    def process_scan(self, scan_msg):
        """
        輸入: sensor_msgs/LaserScan
        輸出: List of [x, y] (相對於 Laser Frame 的障礙物中心點)
        """
        ranges = np.array(scan_msg.ranges)

        # 1. 前處理：過濾掉無限遠或太近的雜訊
        # BARN 的動態障礙物通常在 5米內，太遠的不用管，節省計算
        valid_mask = (ranges > scan_msg.range_min) & (ranges < 5.0)
        valid_ranges = ranges[valid_mask]

        # 取得對應的角度
        angle_min = scan_msg.angle_min
        angle_increment = scan_msg.angle_increment
        valid_indices = np.where(valid_mask)[0]
        valid_angles = angle_min + valid_indices * angle_increment

        # 2. 轉換為笛卡爾座標 (Polar -> Cartesian)
        # x = r * cos(theta), y = r * sin(theta)
        points_x = valid_ranges * np.cos(valid_angles)
        points_y = valid_ranges * np.sin(valid_angles)

        if len(points_x) == 0:
            return []

        points = np.column_stack((points_x, points_y))

        # 3. 執行分群 (Simple Euclidean Clustering)
        centroids = []
        current_cluster = [points[0]]

        for i in range(1, len(points)):
            # 計算當前點與上一個點的距離
            dist = np.linalg.norm(points[i] - points[i-1])

            if dist < self.threshold:
                # 距離夠近，屬於同一個物體
                current_cluster.append(points[i])
            else:
                # 斷開了，結算上一個群集
                if len(current_cluster) >= self.min_points:
                    # 計算群集的幾何中心 (Centroid)
                    cluster_np = np.array(current_cluster)
                    centroid = np.mean(cluster_np, axis=0)
                    centroids.append(centroid.tolist())

                # 開始新的群集
                current_cluster = [points[i]]

        # 處理最後一個群集
        if len(current_cluster) >= self.min_points:
            cluster_np = np.array(current_cluster)
            centroid = np.mean(cluster_np, axis=0)
            centroids.append(centroid.tolist())

        return centroids
