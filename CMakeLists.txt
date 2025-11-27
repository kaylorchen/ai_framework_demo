cmake_minimum_required(VERSION 3.22)
project(ai_framework_demo)

if (NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_BUILD_TYPE "Release") # None, Debug, Release, RelWithDebInfo, MinSizeRel
endif()
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -Wno-unused-parameter -O0 -g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS}  -Wno-unused-parameter -O3 -g -Wall")

find_package(yaml-cpp REQUIRED)
find_package(OpenCV REQUIRED)
find_package(kaylordut REQUIRED)


set(INC ${OpenCV_INCLUDE_DIRS})
set(LINKS ${YAML_CPP_LIBRARIES} ${OpenCV_LIBS} ${kaylordut_LIBS})

find_package(ai_instance_tensorrt QUIET)
if (ai_instance_tensorrt_FOUND)
    message(STATUS "Found ai_instance_tensorrt")
    set(AI_LIB ${ai_instance_tensorrt_LIBS})
    add_definitions(-DTRT=1)
else ()
    find_package(ai_instance_onnx QUIET)
    if (ai_instance_onnx_FOUND)
        message(STATUS "Found ai_instance_onnx")
        set(AI_LIB ${ai_instance_onnx_LIBS})
        add_definitions(-DONNX=1)
    endif ()
endif ()
find_package(ai_instance_rk3588 QUIET)
if (ai_instance_rk3588_FOUND)
    message(STATUS "Found ai_instance_rk3588")
    set(AI_LIB ${ai_instance_rk3588_LIBS})
    add_definitions(-DRK3588=1)
endif ()
find_package(ai_instance_nnrt QUIET)
if (ai_instance_nnrt_FOUND)
    message(STATUS "Found ai_instance_nnrt")
    set(AI_LIB ${ai_instance_nnrt_LIBS})
    add_definitions(-DNNRT=1)
endif ()
message(STATUS "AI_LIB is ${AI_LIB}")


include_directories(./)
include_directories(INC)

add_subdirectory(image_process)
file(GLOB_RECURSE UTILS "utils/*.cpp")

add_executable(yolo_mutilthreading_demo mutilthreading/mutilthreading_demo.cpp ${UTILS})
target_link_libraries(yolo_mutilthreading_demo ${LINKS} ${AI_LIB} yolo_postprocess yolo_preprocess)

add_executable(jialin_demo mutilthreading/jialin_demo.cpp ${UTILS})
target_link_libraries(jialin_demo ${LINKS} ${AI_LIB} yolo_postprocess yolo_preprocess)

add_executable(depth_demo depth_anything/depth_demo.cpp)
target_link_libraries(depth_demo depth_image_process  ${LINKS} ${AI_LIB})

add_executable(depth_camera_demo depth_anything/depth_camera_demo.cpp)
target_link_libraries(depth_camera_demo depth_image_process ${LINKS} ${AI_LIB})

add_executable(inference_test inference_without_preprocess_postprocess/main.cpp)
target_link_libraries(inference_test ${LINKS} ${AI_LIB})

set(PCL_DIR /home/kaylor/work/pcl/1.15.1/share/pcl-1.15)
find_package(PCL 1.15 REQUIRED COMPONENTS)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
message(STATUS "PCL_INCLUDE_DIRS is ${PCL_INCLUDE_DIRS}")
message(STATUS "PCL_LIBRARY_DIRS is ${PCL_LIBRARY_DIRS}")

add_executable(foundation_stereo
        foundation_stereo/main_read_files.cpp
        foundation_stereo/foundation_stereo_image_process.cpp
        foundation_stereo/image_padder.cpp)
target_link_libraries(foundation_stereo ${LINKS} ${AI_LIB} ${PCL_LIBRARIES})

add_executable(foundation_stereo_zed
        foundation_stereo/main_zed.cpp
        foundation_stereo/foundation_stereo_image_process.cpp
        foundation_stereo/image_padder.cpp
        foundation_stereo/display_cloud.cpp
)
target_link_libraries(foundation_stereo_zed ${LINKS} ${AI_LIB} ${PCL_LIBRARIES})
