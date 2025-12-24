cmake_minimum_required(VERSION 3.0.2)
project(hlsd_local_planner)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  nav_core
  base_local_planner
  costmap_2d
  tf2
  tf2_ros
  tf2_geometry_msgs
  geometry_msgs
  sensor_msgs
  pluginlib
)

# 連結 ONNX Runtime (假設已安裝到系統路徑)
# 如果是自定義路徑，請用 include_directories() 和 link_directories()
find_library(ONNXRUNTIME_LIB onnxruntime)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES hlsd_local_planner
  CATKIN_DEPENDS nav_core costmap_2d roscpp tf2_ros
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  # /path/to/onnxruntime/include  <-- 如果沒裝在系統路徑，取消註解這行
)

add_library(hlsd_local_planner src/hlsd_planner.cpp)
target_link_libraries(hlsd_local_planner
  ${catkin_LIBRARIES}
  ${ONNXRUNTIME_LIB}
)

# 安裝與導出 Plugin
install(TARGETS hlsd_local_planner
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
)
install(FILES hlsd_plugin.xml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
