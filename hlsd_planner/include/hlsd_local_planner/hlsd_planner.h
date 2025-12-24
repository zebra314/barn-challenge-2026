#ifndef HLSD_PLANNER_H_
#define HLSD_PLANNER_H_

#include <ros/ros.h>
#include <nav_core/base_local_planner.h>
#include <base_local_planner/odometry_helper_ros.h>
#include <tf2_ros/buffer.h>
#include <costmap_2d/costmap_2d_ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Twist.h>
#include <sensor_msgs/LaserScan.h>

// ONNX Runtime C++ API
#include <onnxruntime_cxx_api.h>

#include <vector>
#include <mutex>
#include <string>

namespace hlsd_local_planner {

class HLSDPlanner : public nav_core::BaseLocalPlanner {
public:
  HLSDPlanner();
  ~HLSDPlanner();

  // nav_core 介面必須實作的三個函數
  void initialize(std::string name, tf2_ros::Buffer* tf, costmap_2d::Costmap2DROS* costmap_ros);
  bool setPlan(const std::vector<geometry_msgs::PoseStamped>& plan);
  bool computeVelocityCommands(geometry_msgs::Twist& cmd_vel);
  bool isGoalReached();

private:
  // --- 輔助函數 ---
  void laserCallback(const sensor_msgs::LaserScan::ConstPtr& msg);
  std::vector<float> preprocess(const sensor_msgs::LaserScan& scan,
                                const geometry_msgs::PoseStamped& robot_pose,
                                const geometry_msgs::Twist& robot_vel);
  bool safetyCheck(const sensor_msgs::LaserScan& scan, float v, float w);

  // --- ROS 變數 ---
  ros::NodeHandle nh_;
  ros::Subscriber laser_sub_;
  tf2_ros::Buffer* tf_;
  costmap_2d::Costmap2DROS* costmap_ros_;
  base_local_planner::OdometryHelperRos odom_helper_;

  std::vector<geometry_msgs::PoseStamped> global_plan_;
  sensor_msgs::LaserScan current_scan_;
  std::mutex scan_mutex_;
  bool scan_received_;
  bool initialized_;

  // --- ONNX Runtime 變數 ---
  Ort::Env ort_env_;
  Ort::Session* session_;
  std::vector<const char*> input_node_names_;
  std::vector<const char*> output_node_names_;

  // --- 配置參數 (Config) ---
  std::string model_path_;
  double max_v_, max_w_;
  double goal_lookahead_dist_; // 往前看多遠找局部目標 (例如 1.5m)
  int input_lidar_dim_;        // 60
  double lidar_max_range_;     // 5.0 or 10.0 (必須與訓練一致)
};

} // namespace hlsd_local_planner

#endif
