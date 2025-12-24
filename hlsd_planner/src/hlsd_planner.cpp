#include "hlsd_local_planner/hlsd_planner.h"
#include <pluginlib/class_list_macros.h>
#include <tf2/utils.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <algorithm>
#include <cmath>

// 註冊為 Plugin
PLUGINLIB_EXPORT_CLASS(hlsd_local_planner::HLSDPlanner, nav_core::BaseLocalPlanner)

namespace hlsd_local_planner {

HLSDPlanner::HLSDPlanner()
  : initialized_(false), scan_received_(false), session_(nullptr),
    ort_env_(ORT_LOGGING_LEVEL_WARNING, "HLSDPlanner") {}

HLSDPlanner::~HLSDPlanner() {
  if (session_) delete session_;
}

void HLSDPlanner::initialize(std::string name, tf2_ros::Buffer* tf, costmap_2d::Costmap2DROS* costmap_ros) {
  if (!initialized_) {
    ros::NodeHandle private_nh("~/" + name);
    nh_ = private_nh;
    tf_ = tf;
    costmap_ros_ = costmap_ros;

    // 1. 載入參數 (必須與 python config 一致)
    private_nh.param("model_path", model_path_, std::string("/tmp/hlsd_model.onnx"));
    private_nh.param("max_v", max_v_, 2.0);
    private_nh.param("max_w", max_w_, 2.0);
    private_nh.param("goal_lookahead_dist", goal_lookahead_dist_, 1.5); // 1.5m
    private_nh.param("input_lidar_dim", input_lidar_dim_, 60);
    private_nh.param("lidar_max_range", lidar_max_range_, 5.0); // 訓練時設多少這裡就設多少

    // 2. 初始化 ROS 訂閱
    laser_sub_ = nh_.subscribe<sensor_msgs::LaserScan>("/front/scan", 1, &HLSDPlanner::laserCallback, this);
    odom_helper_.setOdomTopic("odometry/filtered");

    // 3. 初始化 ONNX Runtime
    try {
      Ort::SessionOptions session_options;
      session_options.SetIntraOpNumThreads(1); // i3 CPU 優化: 單線程通常夠快且省資源
      session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

      session_ = new Ort::Session(ort_env_, model_path_.c_str(), session_options);

      // 設定 I/O 名稱 (根據 pytorch export 時的設定)
      input_node_names_ = {"input"};
      output_node_names_ = {"output"};

      ROS_INFO("HLSD: ONNX Model loaded successfully from %s", model_path_.c_str());
    } catch (const Ort::Exception& e) {
      ROS_ERROR("HLSD: Failed to load ONNX model: %s", e.what());
    }

    initialized_ = true;
  }
}

void HLSDPlanner::laserCallback(const sensor_msgs::LaserScan::ConstPtr& msg) {
  std::lock_guard<std::mutex> lock(scan_mutex_);
  current_scan_ = *msg;
  scan_received_ = true;
}

bool HLSDPlanner::setPlan(const std::vector<geometry_msgs::PoseStamped>& plan) {
  if (!initialized_) return false;
  global_plan_ = plan;
  return true;
}

// === 核心: 每一幀的控制迴圈 ===
bool HLSDPlanner::computeVelocityCommands(geometry_msgs::Twist& cmd_vel) {
  if (!initialized_ || !scan_received_) {
    ROS_WARN_THROTTLE(1.0, "HLSD: Waiting for initialization or laser scan...");
    return false;
  }

  // 1. 獲取機器人狀態
  tf2::Stamped<tf2::Transform> robot_tf;
  geometry_msgs::PoseStamped robot_pose;
  costmap_ros_->getRobotPose(robot_pose);

  nav_msgs::Odometry base_odom;
  odom_helper_.getOdom(base_odom);
  geometry_msgs::Twist robot_vel = base_odom.twist.twist;

  // 2. 獲取最新的 Scan (Thread-safe)
  sensor_msgs::LaserScan scan_copy;
  {
    std::lock_guard<std::mutex> lock(scan_mutex_);
    scan_copy = current_scan_;
  }

  // 3. 資料前處理 (Preprocessing) -> Tensor
  // 這裡回傳的是 normalized 的 [Lidar(60), Goal(2), Vel(2)] = 64
  std::vector<float> input_tensor_values = preprocess(scan_copy, robot_pose, robot_vel);

  // 4. ONNX 推理 (Inference)
  std::vector<int64_t> input_shape = {1, 64}; // Batch size = 1

  try {
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_tensor_values.data(), input_tensor_values.size(),
        input_shape.data(), input_shape.size());

    auto output_tensors = session_->Run(
        Ort::RunOptions{nullptr},
        input_node_names_.data(), &input_tensor, 1,
        output_node_names_.data(), 1);

    float* output_data = output_tensors.front().GetTensorMutableData<float>();

    // 5. 後處理 (Post-processing)
    // 模型的輸出通常是真實物理值 (依據 train.py 的 loss 計算方式)
    // 如果您在 train 時把 label 也歸一化了，這裡要乘回 MAX_V。
    // *假設*: train.py 直接 regression 真實速度。
    float target_v = output_data[0];
    float target_w = output_data[1];

    // 限制範圍 (Clip)
    target_v = std::max(0.0f, std::min(target_v, (float)max_v_));
    target_w = std::max((float)-max_w_, std::min(target_w, (float)max_w_));

    // 6. Safety Filter (最後防線)
    if (!safetyCheck(scan_copy, target_v, target_w)) {
        ROS_WARN_THROTTLE(0.5, "HLSD: Safety Filter Triggered! Stopping.");
        target_v = 0.0;
        // target_w = 0.0; // 可以允許原地旋轉，或也設為0
    }

    cmd_vel.linear.x = target_v;
    cmd_vel.angular.z = target_w;

    return true;

  } catch (const Ort::Exception& e) {
    ROS_ERROR("HLSD: Inference Error: %s", e.what());
    return false;
  }
}

// === 前處理邏輯 (必須與 Python 完全一致) ===
std::vector<float> HLSDPlanner::preprocess(const sensor_msgs::LaserScan& scan,
                                           const geometry_msgs::PoseStamped& robot_pose,
                                           const geometry_msgs::Twist& robot_vel) {
  std::vector<float> feature_vector;
  feature_vector.reserve(64);

  // A. LiDAR Min-Pooling (720 -> 60)
  int chunk_size = scan.ranges.size() / input_lidar_dim_;

  for (int i = 0; i < input_lidar_dim_; ++i) {
    float min_val = lidar_max_range_;
    for (int j = 0; j < chunk_size; ++j) {
      int idx = i * chunk_size + j;
      if (idx >= scan.ranges.size()) break;

      float r = scan.ranges[idx];
      // 處理 inf / nan
      if (std::isinf(r) || std::isnan(r)) r = lidar_max_range_;
      if (r < min_val) min_val = r;
    }
    // 歸一化 [0, 1]
    feature_vector.push_back(std::min(min_val, (float)lidar_max_range_) / lidar_max_range_);
  }

  // B. Local Goal (Relative Coordinates)
  // 在 global plan 中尋找前方 lookahead_dist 處的點
  geometry_msgs::PoseStamped target_pose = global_plan_.back(); // 預設為終點

  // 簡單的搜尋邏輯: 找第一個距離大於 lookahead 的點
  // 注意: 這裡需要更嚴謹的 frame transform (map -> base_link)
  // 為簡化代碼，假設 global_plan 已經被 transform 到 odom，我們需要算 base_link 下的座標

  for (const auto& pose : global_plan_) {
    double dist = std::hypot(pose.pose.position.x - robot_pose.pose.position.x,
                             pose.pose.position.y - robot_pose.pose.position.y);
    if (dist >= goal_lookahead_dist_) {
      target_pose = pose;
      break;
    }
  }

  // Transform target_pose to base_link
  geometry_msgs::PoseStamped local_target;
  try {
    // 等待 transform
    // 注意: computeVelocityCommands 頻率高，這裡最好不要 lookupTransform 導致阻塞
    // 實作中應使用 cached transform
    geometry_msgs::TransformStamped transform = tf_->lookupTransform("base_link", target_pose.header.frame_id, ros::Time(0));
    tf2::doTransform(target_pose, local_target, transform);
  } catch (tf2::TransformException &ex) {
    ROS_WARN("HLSD: Transform failed %s", ex.what());
    local_target.pose.position.x = goal_lookahead_dist_; // Fallback
    local_target.pose.position.y = 0;
  }

  double dx = local_target.pose.position.x;
  double dy = local_target.pose.position.y;
  double dist = std::sqrt(dx*dx + dy*dy);
  double angle = std::atan2(dy, dx);

  // Goal Normalization (除以 3.0m, PI)
  feature_vector.push_back(std::min(dist / 3.0, 1.0));
  feature_vector.push_back(angle / M_PI);

  // C. Last Velocity (Normalization)
  // 假設 max_v = 2.0, max_w = 2.0
  feature_vector.push_back(robot_vel.linear.x / max_v_);
  feature_vector.push_back(robot_vel.angular.z / max_w_);

  return feature_vector;
}

bool HLSDPlanner::safetyCheck(const sensor_msgs::LaserScan& scan, float v, float w) {
  // 簡單的緊急剎車: 如果正前方 0.3m 有障礙物且速度 > 0.1
  // 這是一個非常基礎的實作，建議在複雜場景增強它
  if (v <= 0.1) return true; // 已經很慢了，允許微調

  int center_idx = scan.ranges.size() / 2;
  int width = 40; // 檢查中心左右各 20 條射線
  float stop_dist = 0.35;

  for (int i = center_idx - width; i < center_idx + width; ++i) {
    if (i >= 0 && i < scan.ranges.size()) {
       float r = scan.ranges[i];
       if (!std::isinf(r) && r < stop_dist) {
         return false; // 危險!
       }
    }
  }
  return true;
}

bool HLSDPlanner::isGoalReached() {
  if (global_plan_.empty()) return true;
  // 檢查與 global_plan 終點的距離
  // ... (標準實作，略)
  return false;
}

} // namespace
