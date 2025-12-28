from dataclasses import dataclass
import numpy as np
from typing import Optional
from matplotlib import pyplot as plt
from enum import Enum, auto

class MotionType(Enum):
    # Basic
    FORWARD = auto()
    TURN = auto()

    # Barn Challenge Specific
    CHICANE = auto()
    THREADING = auto()

    # Recovery
    BRAKE = auto()
    SPIN = auto()
    BACKWARD = auto()

@dataclass
class TrajectoryConfig:
    motion_type: MotionType
    states: list[list[float]]
    commands: list[list[float]]
    dt: float = 0.1


class Trajectory:
    def __init__(self,
                 states: list[list[float]],
                 commands: list[list[float]],
                 motion_type: MotionType,
                 dt: float = 0.1):
        """
        :param states: list of [x, y, theta]
        :param commands: list of [v, w]
        :param motion_type: MotionType
        :param dt: time step between frames
        """
        self.states = np.array(states, dtype=np.float32) # Shape: (N, 3)
        self.commands = np.array(commands, dtype=np.float32) # Shape: (N, 2)
        self.motion_type = motion_type
        self.dt = dt

        assert len(self.states) == len(self.commands), "States and Commands length mismatch!"
        self.length = len(self.states)

    def __len__(self) -> int:
        return self.length

    @property
    def duration(self) -> float:
        return self.length * self.dt

    def get_frame_data(self, current_idx: int, history_steps: int, future_steps: int) -> Optional[tuple]:
        """
        :return: (current_state, past_states, future_states, label_cmd) or None
        """
        start_idx = current_idx - history_steps
        end_idx = current_idx + future_steps

        # Check bounds
        if start_idx < 0 or end_idx >= self.length:
            return None

        # Past, from start to current-1
        past_states = self.states[start_idx : current_idx]

        # Current
        current_state = self.states[current_idx]

        # Future, from current+1 to end
        future_states = self.states[current_idx + 1 : end_idx + 1]

        # Label: current executing command
        label_cmd = self.commands[current_idx]

        return (current_idx, current_state, past_states, future_states, self.commands[current_idx-1], label_cmd)

    def plot(self) -> None:
        """
        Plot the trajectory
        """
        x = self.states[:, 0]
        y = self.states[:, 1]
        theta = self.states[:, 2]

        plt.figure()
        plt.plot(x, y, marker='o')

        # Plot velocity vector at start point
        start_v = self.commands[0, 0]  # linear velocity
        start_dx = start_v * np.cos(theta[0])
        start_dy = start_v * np.sin(theta[0])
        plt.quiver(x[0], y[0], start_dx, start_dy, color='green', scale=20, width=0.005, label='Start velocity')

        # Plot velocity vector at end point
        end_v = self.commands[-1, 0]  # linear velocity
        end_dx = end_v * np.cos(theta[-1])
        end_dy = end_v * np.sin(theta[-1])
        plt.quiver(x[-1], y[-1], end_dx, end_dy, color='red', scale=20, width=0.005, label='End velocity')

        plt.title(f'Trajectory: {self.motion_type.name}')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.axis('equal')
        plt.grid(True)
        plt.legend()
        plt.show()
