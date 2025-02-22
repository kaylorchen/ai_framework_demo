
message(STATUS "Target: ${TARGET_NAME} Definition: ${TARGET_DEFINITION}, Link Libs: ${TARGET_LINK_LIBS} Sources: ${TARGET_SRC}")

add_library(${TARGET_NAME} SHARED ${TARGET_SRC})
target_compile_definitions(${TARGET_NAME} PUBLIC ${TARGET_DEFINITION})
target_link_libraries(${TARGET_NAME} PRIVATE ${TARGET_LINK_LIBS})
if (ONNX)
    target_link_libraries(${TARGET_NAME} PUBLIC ${ORT_LIBS})
endif ()
install(TARGETS ${TARGET_NAME}
        EXPORT ${TARGET_NAME}Targets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(EXPORT ${TARGET_NAME}Targets
        FILE ${TARGET_NAME}Targets.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${TARGET_NAME})

write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}ConfigVersion.cmake"
        VERSION "${version}"
        COMPATIBILITY AnyNewerVersion
)
# create config file
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
        "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}Config.cmake"
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)
# install config files
install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}Config.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}ConfigVersion.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${TARGET_NAME}
)