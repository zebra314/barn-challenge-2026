import numpy as np
from simulation.robot_config import RobotConfig

class JackalSimulator:
    """
    Jackal Robot Simulator
    """
    def __init__(self, config=RobotConfig()):
        self.cfg = config
        self.reset()

    def reset(self):
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.v_actual = 0.0
        self.w_actual = 0.0
        self.time = 0.0

    def step(self, target_v, target_w):
        """
        :return: (pose, actual_vel) -> ((x, y, theta), (v, w))
        """
        dt = self.cfg.DT

        # Inertia
        if target_v > self.v_actual:
            self.v_actual = min(target_v, self.v_actual + self.cfg.ACC_LIM_V * dt)
        else:
            self.v_actual = max(target_v, self.v_actual - self.cfg.ACC_LIM_V * dt)

        if target_w > self.w_actual:
            self.w_actual = min(target_w, self.w_actual + self.cfg.ACC_LIM_W * dt)
        else:
            self.w_actual = max(target_w, self.w_actual - self.cfg.ACC_LIM_W * dt)

        # Actuation Noise
        noisy_v = self.v_actual + np.random.normal(0, self.cfg.NOISE_V_STD)
        noisy_w = self.w_actual + np.random.normal(0, self.cfg.NOISE_W_STD)

        # Under-steering
        slip_ratio = 1.0 - (abs(self.v_actual) / self.cfg.MAX_V) * self.cfg.SLIP_FACTOR_MAX
        effective_w = noisy_w * slip_ratio

        # Lateral Slip
        lateral_v_coeff = 0.05

        # v_x: Forward velocity (Body Frame x)
        # v_y: Lateral velocity (Body Frame y), when turning left (w>0), centrifugal force is to the right (y<0)
        v_x = noisy_v
        v_y = -lateral_v_coeff * noisy_v * effective_w

        # Convert Body Frame velocities (vx, vy) to Global Frame displacements (dx, dy)
        # Rotation Matrix R(theta)
        # [ cos -sin ]
        # [ sin  cos ]
        c = np.cos(self.theta)
        s = np.sin(self.theta)

        # Matrix multiplication expansion
        dx = v_x * c - v_y * s
        dy = v_x * s + v_y * c

        # Update Pose
        self.x += dx * dt
        self.y += dy * dt
        self.theta += effective_w * dt

        # Normalize angle to (-pi, pi]
        self.theta = np.arctan2(np.sin(self.theta), np.cos(self.theta))
        self.time += dt

        # Return State and Actual Velocity (without noise)
        # Note: For training data, the Input Last_Vel is recommended to use self.v_actual (or target_v)
        # Because this is the velocity the robot "thinks" it has, not the true noisy velocity
        return np.array([self.x, self.y, self.theta]), np.array([self.v_actual, self.w_actual])
