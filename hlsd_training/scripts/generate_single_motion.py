import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import logging
from simulation.jackal_simulator import JackalSimulator
from simulation.motion_generator import MotionGenerator
from utils.logging_config import setup_logging

def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    steps = 1000
    out_dir = Path(__file__).resolve().parent.parent / 'data'
    out_dir.mkdir(parents=True, exist_ok=True)

    sim = JackalSimulator()
    generator = MotionGenerator(sim)

    # Run the motion generation
    # states shape: (N, 5) -> x, y, theta, v_act, w_act
    # cmds shape:   (N, 2) -> v_cmd, w_cmd
    states, cmds = generator.generate_single(steps)

    np.save(out_dir / 'raw_states.npy', states)
    np.save(out_dir / 'raw_commands.npy', cmds)

    logger.info(f"\nData saved to {out_dir}")
    logger.info(f"States shape: {states.shape}")
    logger.info(f"Commands shape: {cmds.shape}")

    plt.figure(figsize=(10, 10))
    x = states[:, 0]
    y = states[:, 1]

    v_act = states[:, 3]
    plt.scatter(x, y, c=v_act, cmap='viridis', s=1, alpha=0.5)
    plt.colorbar(label='Velocity (m/s)')
    plt.title(f"Generated Jackal Trajectory ({steps} steps)\nColor represents speed")
    plt.axis('equal')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
