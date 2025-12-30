cmake_minimum_required(VERSION 3.0.2)
project(amcn)

find_package(catkin REQUIRED COMPONENTS
  rospy
  geometry_msgs
  nav_msgs
  tf2_ros
  tf2_geometry_msgs
)

# 1. 啟用 Python 模組安裝 (這會呼叫 setup.py)
catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS rospy geometry_msgs nav_msgs tf2_ros
)

# 2. 安裝 Python 可執行節點
catkin_install_python(PROGRAMS
  src/nodes/mpc_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# 3. 安裝設定檔與 Launch 檔
install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
