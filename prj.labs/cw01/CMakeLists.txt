project(cw01)
add_executable(cw01 cw01.cpp cw01.report.md.in.txt)
target_link_libraries(cw01 ${OpenCV_LIBS})
install(DIRECTORY ${CMAKE_SOURCE_DIR}/data DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
install(DIRECTORY ${CMAKE_SOURCE_DIR}/output DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

file(READ cw01.cpp cpp_source)
# prepare report md file
configure_file(cw01.report.md.in.txt ${CMAKE_CURRENT_BINARY_DIR}/cw01.report.md @ONLY)
add_custom_command(TARGET ${PROJECT_NAME}
  POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_BINARY_DIR}/cw01.report.md" "$<TARGET_FILE_DIR:${PROJECT_NAME}>")