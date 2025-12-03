cmake_minimum_required(VERSION 3.8)
project(hunter_stereo)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(ai_instance_tensorrt QUIET)
if (ai_instance_tensorrt_FOUND)
  message(STATUS "Found ai_instance_tensorrt")
  set(AI_LIB ${ai_instance_tensorrt_LIBS})
  include_directories(/usr/local/cuda/include)
  add_definitions(-DTRT=1)
else ()
  find_package(ai_instance_onnx QUIET)
  if (ai_instance_onnx_FOUND)
    message(STATUS "Found ai_instance_onnx")
    set(AI_LIB ${ai_instance_onnx_LIBS})
    add_definitions(-DONNX=1)
  endif ()
endif ()
find_package(kaylordut REQUIRED)
find_package(OpenCV REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

add_executable(${PROJECT_NAME}_node src/main_ros2.cpp src/stereo_image_process.cpp src/image_padder.cpp)
ament_target_dependencies(${PROJECT_NAME}_node rclcpp sensor_msgs pcl_ros pcl_conversions)
target_link_libraries(${PROJECT_NAME}_node ${AI_LIB} ${OpenCV_LIBS} ${kaylordut_LIBS})
target_include_directories(${PROJECT_NAME}_node PUBLIC ${OpenCV_INCLUDE_DIRS})
install(TARGETS ${PROJECT_NAME}_node
        DESTINATION lib/${PROJECT_NAME})

ament_package()
