message(STATUS "build ai framework library")
include_directories(./)
set(PREFIX_NAME ai_instance)
if (TRT)
    message(STATUS "build ${PREFIX_NAME}_tensorrt library")
    set(TARGET_NAME ${PREFIX_NAME}_tensorrt)
    set(TARGET_DEFINITION TRT)
    set(TARGET_LINK_LIBS ${TRT_LIBS} ${kaylordut_LIBS})
    set(TARGET_SRC
            ai_framework.cpp
            ai_instance.cpp
            platform/tensorrt/tensorrt.cpp
    )
    include(target.cmake)
    install(FILES platform/tensorrt/tensorrt.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/platform/tensorrt)
endif ()
if (ONNX)
    message(STATUS "build ${PREFIX_NAME}_onnx library")
    set(TARGET_NAME ${PREFIX_NAME}_onnx)
    set(TARGET_DEFINITION ONNX)
    set(TARGET_LINK_LIBS ${ORT_LIBS} ${kaylordut_LIBS})
    set(TARGET_SRC
            ai_framework.cpp
            ai_instance.cpp
            platform/onnxruntime/onnxruntime.cpp
    )
    include(target.cmake)
    install(FILES platform/onnxruntime/onnxruntime.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/platform/onnxruntime)
endif ()
if (NNRT)
    message(STATUS "build ${PREFIX_NAME}_nnrt library")
    set(TARGET_NAME ${PREFIX_NAME}_nnrt)
    set(TARGET_DEFINITION NNRT)
    set(TARGET_LINK_LIBS ${kaylordut_LIBS} ascendcl)
    set(TARGET_SRC
            ai_framework.cpp
            ai_instance.cpp
            platform/nnrt/nnrt.cpp
    )
    include(target.cmake)
    install(FILES platform/nnrt/nnrt.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/platform/nnrt)
endif ()
if (RK3588)
    message(STATUS "build ${PREFIX_NAME}_rk3588 library")
    set(TARGET_NAME ${PREFIX_NAME}_rk3588)
    set(TARGET_DEFINITION RK3588)
    set(TARGET_LINK_LIBS ${kaylordut_LIBS} rknnrt)
    set(TARGET_SRC
            ai_framework.cpp
            ai_instance.cpp
            platform/rockchip/rk3588.cpp
    )
    include(target.cmake)
    install(FILES platform/rockchip/rk3588.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/platform/rockchip)
endif ()

install(FILES ai_instance.h ai_framework.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

set(AI_LIB ${TARGET_NAME} PARENT_SCOPE)