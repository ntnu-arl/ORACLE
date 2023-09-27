cmake_minimum_required(VERSION 3.0.2)
project(deep_collision_predictor)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  lmf_planner_srvs_msgs
  voxblox_msgs
)

catkin_package(
  CATKIN_DEPENDS rospy std_msgs
)

###########
## Build ##
###########

include_directories(
  ${catkin_INCLUDE_DIRS}
)

install(PROGRAMS
   evaluate_planner.py
   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)