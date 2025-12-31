import numpy as np

class SimpleTracker:
    def __init__(self):
        self.tracks = {} # 儲存 {id: {'pos': [x, y], 'vel': [vx, vy], 'history': []}}
        self.next_id = 0

    def update(self, detected_centroids, dt):
        """
        detected_centroids: 本次雷達分群後得到的中心點列表 [[x1,y1], [x2,y2]...]
        dt: 兩次掃描的時間差
        """
        # 1. 簡單匹配 (Nearest Neighbor)
        # 對每個新偵測點，找最近的舊軌跡
        assignments = []
        for i, det in enumerate(detected_centroids):
            best_dist = 1.0 # 匹配閾值 (例如 1公尺內)
            best_id = -1

            for tid, track in self.tracks.items():
                # 預測該軌跡現在應該在哪 (Constant Velocity Model)
                pred_pos = track['pos'] + track['vel'] * dt
                dist = np.linalg.norm(det - pred_pos)

                if dist < best_dist:
                    best_dist = dist
                    best_id = tid

            assignments.append((i, best_id))

        # 2. 更新狀態
        new_tracks = {}
        used_ids = set()

        for det_idx, track_id in assignments:
            current_pos = np.array(detected_centroids[det_idx])

            if track_id != -1 and track_id not in used_ids:
                # 匹配成功：更新速度與位置
                prev_pos = self.tracks[track_id]['pos']
                # 簡單速度計算 (可以加個低通濾波平滑一下)
                new_vel = (current_pos - prev_pos) / dt

                new_tracks[track_id] = {
                    'pos': current_pos,
                    'vel': new_vel,
                    'history': self.tracks[track_id]['history'] + [current_pos]
                }
                used_ids.add(track_id)
            else:
                # 沒匹配到：視為新障礙物
                new_tracks[self.next_id] = {
                    'pos': current_pos,
                    'vel': np.array([0.0, 0.0]), # 初始速度設為 0
                    'history': [current_pos]
                }
                self.next_id += 1

        self.tracks = new_tracks
        return self.extract_mpc_params()

    def extract_mpc_params(self):
        # 轉換成 CasADi Solver 需要的矩陣格式 [x, y, vx, vy, r]
        obs_list = []
        for tid, track in self.tracks.items():
            obs_list.append([
                track['pos'][0], track['pos'][1],
                track['vel'][0], track['vel'][1],
                0.3 # 假設半徑 (包含膨脹)
            ])
        return obs_list
