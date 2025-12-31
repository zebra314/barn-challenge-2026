# Week 1

2025-12-27 to 2026-01-02

## Todos

[x] Switch from HSLD to Optimization-based approach (MPC/MPCC + CBF)

[x] System Architecture

[x] move_base (Global Planner) + mpc_node.py (Local Planner)

[x] Stable Path Tracking

[x] RK4 Kinematics + Skid-Steer Correction (omega_gain)

[x] CasADi/IPOPT implementation with Manual Yaw Calculation (Unwrap)

[x] Fix DynaBARN global costmap bounds & namespace mismatch

[] Perception & Avoidance

[] Implement lidar_clustering.py & obstacle_tracker.py

[] Formulate obstacle constraints (Soft Constraints + Slack variables)

## Summary

Architecture: Established a robust workflow where move_base handles global planning (Navfn) and a custom Python MPC node handles local control, solving previous ROS plugin limitations.

Control: Achieved stable trajectory tracking. Resolved critical Sim-to-Real issues including "Skid-Steer under-steering" (via phenomenological gain) and "Yaw discontinuity" (via manual path unwrapping).

Navigation Fixes: Rectified Global Costmap configurations (rolling_window, dimensions) to support long-range goals in DynaBARN environments.

Next Step: Transitioning from pure tracking to obstacle avoidance by integrating LiDAR clustering data into the CasADi solver.
