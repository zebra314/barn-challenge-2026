# Week 0

2025-12-20 to 2025-12-26

## Todos

- [] Setup the environment
- [] Create the wrapper for new local planner
    - hlsd_local_planner
    - in c++
    - inherit from nav_core::BaseLocalPlanner
- [] Train the hlsd_model
    - hlsd_model.onnx
- [] Import the hlsd_model to new local planner

## Summary

- Global Planner
    - Plan the general path to the goal
    - Memory the dead ends

- Local Planner
    - Use the model to predict the cmd_vel with input of laser scan, goal point and current velocity
    - Safety Filter

- Velocity Smoother
    - smoother.yaml

- In current launch file, replace the TrajectoryPlannerROS to local planner with hlsd_local_planner
