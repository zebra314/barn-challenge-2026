from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum, auto

class MotionType(Enum):
    # Basic
    FORWARD = auto()
    TURN = auto()

    # Barn Challenge Specific
    OPTIMAL = auto()
    CHICANE = auto()
    THREADING = auto()

    # Recovery
    BRAKE = auto()
    SPIN = auto()
    BACKWARD = auto()

@dataclass
class Obstacle:
    position: Tuple[float, float]
    radius: float

@dataclass
class Trajectory:
    states: list[list[float]]
    commands: list[list[float]]

    def plot(self) -> None:
        pass

@dataclass
class ScenarioConfig:
    # Basic
    motion_type: MotionType
    dt: float = 0.1
    seed: Optional[int] = None

    # Simulation
    robot_width: float = 0.5
    robot_length: float = 0.7

    max_linear_velocity: float = 2.0
    max_linear_acceleration: float = 1.0

    max_angular_velocity: float = 2.0
    max_angular_acceleration: float = 3.0

    std_linear_noise: float = 0.02
    std_angular_noise: float = 0.05

    max_slip_factor: float = 0.2

    # Trajectory

    # Obstacle
    obstacles


class Scenario:
    """
    Wrapper class that initializes the pipeline parameters.
    """
    def __init__(self, config: ScenarioConfig):
        self.config = config
        self.trajectory = None
        self.hallucination = None

    def generate(self):
        """
        Orchestrates the generation process based on the config.
        """
        # 1. Generate Trajectory
        self.trajectory = self.generate_trajectory(self.config)

        # 2. Generate Hallucination/Environment based on trajectory
        self.hallucination = self._generate_hallucination(
            self.config.halluc_params,
            self.trajectory
        )

        return self.trajectory, self.hallucination

    def _generate_trajectory(self, params: TrajectoryParams):
        # Placeholder for your actual trajectory generator logic
        # e.g., return TrajectoryGenerator.create(params)
        print(f"Generating {params.motion_type.name} trajectory with speed {params.speed}...")
        return {"type": params.motion_type, "data": []}

    def _generate_hallucination(self, params: HallucinationParams, trajectory):
        # Placeholder for your actual hallucination generator logic
        # e.g., return HallucinationGenerator.create(params, trajectory)
        print(f"Generating hallucinations with density {params.obstacle_density}...")
        return {"obstacles": []}

# Example Factory for common scenarios
class ScenarioFactory:
    @staticmethod
    def create_chicane_scenario() -> Scenario:
        return Scenario(ScenarioConfig(
            name="Classic Chicane",
            traj_params=TrajectoryParams(
                motion_type=MotionType.chicane,
                speed=1.5,
                duration=8.0
            ),
            halluc_params=HallucinationParams(
                obstacle_density=0.5,
                hallucination_type="static_pillars"
            )
        ))

    @staticmethod
    def create_fast_forward_scenario() -> Scenario:
        return Scenario(ScenarioConfig(
            name="Fast Forward",
            traj_params=TrajectoryParams(
                motion_type=MotionType.forward,
                speed=3.0
            ),
            halluc_params=HallucinationParams(
                obstacle_density=0.1
            )
        ))
