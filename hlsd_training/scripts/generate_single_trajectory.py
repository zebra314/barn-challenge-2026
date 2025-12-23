from simulation.trajectory_generator import TrajectoryGenerator
from simulation.jackal_simulator import JackalSimulator
from common.motion_type import MotionType

if __name__ == "__main__":

    sim = JackalSimulator()
    traj_gen = TrajectoryGenerator(sim)
    traj = traj_gen.generate_single(MotionType.turn)
    print(f"Generated trajectory with {len(traj)} frames, duration: {traj.duration} seconds.")

    traj.plot()
