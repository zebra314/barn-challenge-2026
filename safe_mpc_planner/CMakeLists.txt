cmake_minimum_required(VERSION 3.0.2)
project(safe_mpc_planner)

find_package(catkin REQUIRED COMPONENTS
  rospy
  geometry_msgs
  nav_msgs
  tf2_ros
  tf2_geometry_msgs
)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS rospy geometry_msgs nav_msgs tf2_ros
)

file(GLOB NODE_PY_FILES "${CMAKE_CURRENT_SOURCE_DIR}/python/nodes/*.py")
catkin_install_python(PROGRAMS
  ${NODE_PY_FILES}
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
