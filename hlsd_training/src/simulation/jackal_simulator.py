import numpy as np
from common.robot_config import RobotConfig

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
        self.actual_linear_vel = 0.0
        self.actual_angular_vel = 0.0
        self.time = 0.0

    def step(self, target_v, target_w):
        """
        :return: (pose, actual_vel) -> ((x, y, theta), (v, w))
        """
        dt = self.cfg.dt

        # Inertia
        if target_v > self.actual_linear_vel:
            self.actual_linear_vel = min(target_v, self.actual_linear_vel + self.cfg.max_linear_acc * dt)
        else:
            self.actual_linear_vel = max(target_v, self.actual_linear_vel - self.cfg.max_linear_acc * dt)

        if target_w > self.actual_angular_vel:
            self.actual_angular_vel = min(target_w, self.actual_angular_vel + self.cfg.max_angular_acc * dt)
        else:
            self.actual_angular_vel = max(target_w, self.actual_angular_vel - self.cfg.max_angular_acc * dt)

        # Actuation Noise
        noisy_v = self.actual_linear_vel + np.random.normal(0, self.cfg.std_linear_noise)
        noisy_w = self.actual_angular_vel + np.random.normal(0, self.cfg.std_angular_noise)

        # Under-steering
        slip_ratio = 1.0 - (abs(self.actual_linear_vel) / self.cfg.max_linear_vel) * self.cfg.max_slip_factor
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
        # Note: For training data, the Input Last_Vel is recommended to use self.actual_linear_vel (or target_v)
        # Because this is the velocity the robot "thinks" it has, not the true noisy velocity
        return np.array([self.x, self.y, self.theta]), np.array([self.actual_linear_vel, self.actual_angular_vel])
