cmake_minimum_required(VERSION 3.0.2)
project(team04)

## Uncomment if you add C++ nodes later
# add_compile_options(-std=c++11)

###################################
## Find catkin and dependencies
###################################
find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  nav_msgs
  sensor_msgs
  tf2_ros
  message_generation
  message_runtime
)

###################################
## Package configuration
###################################
catkin_package(
  CATKIN_DEPENDS rospy std_msgs geometry_msgs nav_msgs sensor_msgs tf2_ros
)

include_directories(${catkin_INCLUDE_DIRS})

###################################
## Automatically install Python scripts
###################################
# Find all Python scripts in /scripts and /tools
file(GLOB SCRIPT_FILES
  "${CMAKE_CURRENT_SOURCE_DIR}/scripts/*.py"
  "${CMAKE_CURRENT_SOURCE_DIR}/tools/*.py"
)

# Make them executable
foreach(script ${SCRIPT_FILES})
  file(RELATIVE_PATH script_name ${CMAKE_CURRENT_SOURCE_DIR} ${script})
  message(STATUS "Installing Python script: ${script_name}")
endforeach()

# Install all scripts to the Catkin bin destination
catkin_install_python(PROGRAMS ${SCRIPT_FILES}
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

###################################
## Install additional resource directories
###################################
install(DIRECTORY
  launch
  maps
  worlds
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  PATTERN ".git" EXCLUDE
)

###################################
## (Optional) Testing section
###################################
# catkin_add_nosetests(test)