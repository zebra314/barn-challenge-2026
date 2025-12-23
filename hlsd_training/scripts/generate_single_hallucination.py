from simulation.trajectory_generator import TrajectoryGenerator
from simulation.jackal_simulator import JackalSimulator
from common.motion_type import MotionType
from hallucination.hallucination_generator import HallucinationGenerator

if __name__ == "__main__":

    sim = JackalSimulator()
    traj_gen = TrajectoryGenerator(sim)
    traj = traj_gen.generate_single(MotionType.forward)
    print(f"Generated trajectory with {len(traj)} frames, duration: {traj.duration} seconds.")

    traj.plot()

    halluc_gen = HallucinationGenerator()
    hallucinations = halluc_gen.process_trajectory(traj)
    print(f"Generated {len(hallucinations)} hallucination frames.")
