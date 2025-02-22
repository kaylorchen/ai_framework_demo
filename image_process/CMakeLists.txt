find_package(bytetrack REQUIRED)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_library(yolo_preprocess yolo/yolo_preprocess.cpp)
target_link_libraries(yolo_preprocess PRIVATE ${OpenCV_LIBS})
add_library(yolo_postprocess yolo/yolo_postprocess.cpp)
target_link_libraries(yolo_postprocess PRIVATE ${bytetrack_LIBS} ${OpenCV_LIBS})

add_library(depth_image_process depth_anything/depth_imageprocess.cpp)
target_link_libraries(depth_image_process ${OpenCV_LIBS})
target_compile_definitions(depth_image_process PUBLIC ${TARGET_DEFINITION})