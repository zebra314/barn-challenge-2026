from enum import Enum
import numpy as np
import logging
from simulation.robot_config import RobotConfig
from simulation.jackal_simulator import JackalSimulator

class DrivingMode(Enum):
    CRUISE = 1   # High speed, low turning, simulating long corridors
    TURN = 2     # Medium speed, high turning, simulating avoidance
    SPIN = 3     # Low speed, very high turning, simulating on-the-spot escape

class MotionGenerator:
    """
    Responsible for generating diverse driving commands, simulating human or expert operations in different scenarios.
    """
    def __init__(self, simulator: JackalSimulator):
        self.logger = logging.getLogger(__name__)
        self.sim = simulator
        self.current_mode = DrivingMode.CRUISE
        self.mode_duration = np.random.randint(10, 40)
        self.mode_timer = 0
        self.target_v = 0.0
        self.target_w = 0.0

    def switch_mode(self):
        """
        Randomly switch driving modes to ensure data diversity
        """
        rand = np.random.rand()
        if rand < 0.6:
            self.current_mode = DrivingMode.CRUISE
            self.mode_duration = np.random.randint(10, 40) # Duration 1~4 seconds
        elif rand < 0.9:
            self.current_mode = DrivingMode.TURN
            self.mode_duration = np.random.randint(10, 30) # Duration 1~3 seconds
        else:
            self.current_mode = DrivingMode.SPIN
            self.mode_duration = np.random.randint(5, 20)  # Duration 0.5~2 seconds

        self.mode_timer = 0

    def sample_command(self):
        """
        Generate target velocities based on the current mode
        """
        self.mode_timer += 1
        if self.mode_timer > self.mode_duration:
            self.switch_mode()

        if self.current_mode == DrivingMode.CRUISE:
            # Cruise: mainly straight, occasional slight adjustments
            self.target_v = np.random.uniform(0.5, RobotConfig.MAX_V)
            self.target_w = np.random.uniform(-0.5, 0.5)

        elif self.current_mode == DrivingMode.TURN:
            # Turn: slow down, sharp turns
            self.target_v = np.random.uniform(0.2, 1.2)
            self.target_w = np.random.uniform(-1.5, 1.5)

        elif self.current_mode == DrivingMode.SPIN:
            # On-the-spot spin: very low speed or stop, very high turning (key for BARN escape)
            self.target_v = np.random.uniform(-0.1, 0.1) # Allow slight reverse
            self.target_w = np.random.uniform(-RobotConfig.MAX_W, RobotConfig.MAX_W)

        # Smooth target command changes (to avoid abrupt jumps)
        # This is just to make the generated Label (Command)
        return self.target_v, self.target_w

    def generate_single(self, total_steps) -> tuple[np.ndarray, np.ndarray]:
        history_states = []    # [x, y, theta, v_act, w_act]
        history_commands = []  # [v_cmd, w_cmd]

        self.logger.info(f"Generating {total_steps} steps of motion data...")

        for i in range(total_steps):
            # Decide how the brain wants to drive (Label)
            tv, tw = self.sample_command()

            # Physical simulation execution (Input source)
            state_pose, state_vel = self.sim.step(tv, tw)

            # Record data
            # State: x, y, theta, v_actual, w_actual
            current_full_state = np.concatenate([state_pose, state_vel])

            history_states.append(current_full_state)
            history_commands.append([tv, tw])

            if i % 1000 == 0:
                self.logger.info(f"Progress: {i}/{total_steps}")

        return np.array(history_states), np.array(history_commands)

    def generate_episode(self, total_episodes=200, steps_per_episode=1000) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate multiple independent episodes, inserting NaN between them for later segmentation.
        """
        all_states = []
        all_commands = []

        # Setup NaN padding between episodes
        padding_len = 20
        nan_padding_states = np.full((padding_len, 5), np.nan)
        nan_padding_cmds = np.full((padding_len, 2), np.nan)

        self.logger.info(f"Generating {total_episodes} episodes x {steps_per_episode} steps...")

        for ep in range(total_episodes):
            self.sim.reset()

            # Start each episode in CRUISE mode
            self.current_mode = DrivingMode.CRUISE
            self.mode_timer = 0

            # Store single episode data
            ep_states = []
            ep_commands = []

            for i in range(steps_per_episode):
                # Decide how the brain wants to drive (Label)
                tv, tw = self.sample_command()

                # Physical simulation execution (Input source)
                state_pose, state_vel = self.sim.step(tv, tw)

                # Record data
                # State: x, y, theta, v_actual, w_actual
                current_full_state = np.concatenate([state_pose, state_vel])

                ep_states.append(current_full_state)
                ep_commands.append([tv, tw])

            # Add episode data to all
            all_states.append(np.array(ep_states))
            all_commands.append(np.array(ep_commands))

            # Add NaN padding
            all_states.append(nan_padding_states)
            all_commands.append(nan_padding_cmds)

            if (ep + 1) % 10 == 0:
                self.logger.info(f"Progress: Episode {ep + 1}/{total_episodes}")

        # Concatenate into a single large array
        final_states = np.concatenate(all_states, axis=0)
        final_commands = np.concatenate(all_commands, axis=0)

        self.logger.info(f"Generation complete. Final shape: {final_states.shape}")
        return final_states, final_commands
