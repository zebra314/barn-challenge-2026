import numpy as np
from amcn.common.enums import Scene

class SimpleTracker:
    def __init__(self, dt_threshold=1.0):
        # Store {id: {'state': [x, y, theta, a, b], 'vel': [vx, vy, omega], 'age': 0}}
        self.tracks = {}
        self.next_id = 0
        self.dt_threshold = dt_threshold # If dt is too large (frame drop), reset velocity calculation

        # Smoothing coefficients (0.0~1.0), smaller is smoother but with higher delay
        self.alpha_pos = 0.3
        self.alpha_shape = 0.2  # Shape usually changes little, can use heavier filtering

        self.current_scene = Scene.STATIC_OPEN
        self.consecutive_dynamic_frames = 0
        self.dynamic_trigger_thresh = 8
        self.motion_threshold = 0.5
        self.max_reliable_range = 3.0

    def update(self, detected_obstacles, dt):
        """
        detected_obstacles: List of [x, y, theta, a, b] (from LidarCluster)
        dt: time difference
        """
        if dt > self.dt_threshold:
            # If time difference is too large (e.g., just started), do not calculate velocity
            dt = 0.0

        # 1. Simple matching (Nearest Neighbor based on Position)
        # Although shape is available, matching mainly relies on position distance
        assignments = []

        # Build a list of predicted positions for existing tracks to speed up matching
        track_ids = list(self.tracks.keys())
        predicted_positions = []
        for tid in track_ids:
            track = self.tracks[tid]
            # Constant Velocity Model prediction
            pred_x = track['state'][0] + track['vel'][0] * dt
            pred_y = track['state'][1] + track['vel'][1] * dt
            predicted_positions.append(np.array([pred_x, pred_y]))

        # Start matching each newly detected obstacle
        for i, det in enumerate(detected_obstacles):
            det_pos = np.array(det[:2]) # Extract x, y

            best_dist = 1.0 # Matching threshold (meters)
            best_track_idx = -1

            for idx, pred_pos in enumerate(predicted_positions):
                dist = np.linalg.norm(det_pos - pred_pos)
                if dist < best_dist:
                    best_dist = dist
                    best_track_idx = idx

            # Record (detection index, matched Track ID)
            matched_id = track_ids[best_track_idx] if best_track_idx != -1 else -1
            assignments.append((i, matched_id))

        # 2. Update states
        new_tracks = {}
        used_ids = set()

        for det_idx, track_id in assignments:
            # Extract current measurement: [x, y, theta, a, b]
            meas_state = np.array(detected_obstacles[det_idx])

            # If matched to an existing track and the ID has not been taken by another detection this time
            if track_id != -1 and track_id not in used_ids:
                prev_track = self.tracks[track_id]
                prev_state = prev_track['state']
                prev_vel = prev_track['vel']

                # --- A. Handle angle ambiguity (crucial!) ---
                # The theta calculated by PCA may flip 180 degrees (because the ellipse is symmetric)
                # We want to ensure the new theta is continuous with the old theta
                meas_theta = meas_state[2]
                prev_theta = prev_state[2]

                # Try original angle, +180, -180, see which is closest to the previous one
                candidates = [meas_theta, meas_theta + np.pi, meas_theta - np.pi]
                diffs = [abs(self._normalize_angle(c - prev_theta)) for c in candidates]
                best_theta_idx = np.argmin(diffs)
                corrected_theta = candidates[best_theta_idx]

                # Normalize back to -pi ~ pi
                corrected_theta = self._normalize_angle(corrected_theta)
                meas_state[2] = corrected_theta

                # --- B. Calculate velocity (Finite Difference + Low Pass Filter) ---
                if dt > 0.001:
                    raw_vx = (meas_state[0] - prev_state[0]) / dt
                    raw_vy = (meas_state[1] - prev_state[1]) / dt

                    raw_speed = np.sqrt(raw_vx**2 + raw_vy**2)
                    if raw_speed < 0.2:
                        raw_vx = 0.0
                        raw_vy = 0.0
                        raw_omega = 0.0

                    # Angular velocity
                    angle_diff = self._normalize_angle(corrected_theta - prev_theta)
                    raw_omega = angle_diff / dt

                    # Low pass filter
                    new_vx = (1 - self.alpha_pos) * prev_vel[0] + self.alpha_pos * raw_vx
                    new_vy = (1 - self.alpha_pos) * prev_vel[1] + self.alpha_pos * raw_vy
                    new_omega = (1 - self.alpha_pos) * prev_vel[2] + self.alpha_pos * raw_omega
                else:
                    new_vx, new_vy, new_omega = 0.0, 0.0, 0.0

                # --- C. Smooth shape (a, b) ---
                # Avoid ellipse size fluctuating
                new_a = (1 - self.alpha_shape) * prev_state[3] + self.alpha_shape * meas_state[3]
                new_b = (1 - self.alpha_shape) * prev_state[4] + self.alpha_shape * meas_state[4]

                # Combine new state
                updated_state = [meas_state[0], meas_state[1], corrected_theta, new_a, new_b]

                new_tracks[track_id] = {
                    'state': updated_state,
                    'vel': [new_vx, new_vy, new_omega],
                    'age': prev_track['age'] + 1
                }
                used_ids.add(track_id)

            else:
                # Newly discovered obstacle (initialization)
                new_tracks[self.next_id] = {
                    'state': meas_state.tolist(), # [x, y, theta, a, b]
                    'vel': [0.0, 0.0, 0.0],       # [vx, vy, omega]
                    'age': 0
                }
                self.next_id += 1

        self.tracks = new_tracks

        if self.current_scene in [Scene.DYNAMIC_OPEN, Scene.DYNAMIC_CROWD]:
            return self.extract_mpc_params()

        frame_has_motion = False
        for tid, track in self.tracks.items():
            v_mag = np.linalg.norm(track['vel'][:2])
            dist = np.linalg.norm(track['state'][:2])
            age = track['age']

            obs_a = track['state'][3]
            obs_b = track['state'][4]

            major_axis = max(obs_a, obs_b) * 2.0
            minor_axis = min(obs_a, obs_b) * 2.0

            aspect_ratio = major_axis / max(minor_axis, 0.05)
            is_wall_shape = (aspect_ratio > 4.0) and (major_axis > 0.5)

            not_normal_size = (major_axis > 1.2 or minor_axis < 0.2)

            if not_normal_size or is_wall_shape:
                continue

            if dist < self.max_reliable_range and v_mag > self.motion_threshold and age >= 15 and v_mag < 3.5:
                frame_has_motion = True
                print("Detected motion: Track ID {}, Speed {:.2f} m/s, Distance {:.2f} m".format(tid, v_mag, dist))
                break

        if frame_has_motion:
            self.consecutive_dynamic_frames += 1
        else:
            self.consecutive_dynamic_frames = max(0, self.consecutive_dynamic_frames - 1)

        if self.consecutive_dynamic_frames >= self.dynamic_trigger_thresh:
            self.current_scene = Scene.DYNAMIC_OPEN
            print("Switching to DYNAMIC_OPEN scene mode.")

        return self.extract_mpc_params()

    def extract_mpc_params(self):
        """
        Output format for MPC: [x, y, theta, a, b, vx, vy]
        Note: I choose not to output omega to MPC here because the noise is usually large,
        and most obstacles in BARN move by translation.
        """
        obs_list = []
        is_dynamic = (self.current_scene in [Scene.DYNAMIC_OPEN, Scene.DYNAMIC_CROWD])
        for tid, track in self.tracks.items():
            s = track['state']

            # Ignore omega for obstacle tracking output
            v = track['vel'] if is_dynamic else [0.0, 0.0]

            # [x, y, theta, a, b, vx, vy]
            obs_list.append([s[0], s[1], s[2], s[3], s[4], v[0], v[1]])
        return obs_list

    def _normalize_angle(self, angle):
        """ Normalize angle to [-pi, pi] """
        return (angle + np.pi) % (2 * np.pi) - np.pi
