project(task00)
add_executable(task00 task00.cpp task00.report.md.in.txt)
target_link_libraries(task00 ${OpenCV_LIBS})
install(DIRECTORY ${CMAKE_SOURCE_DIR}/data DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
install(DIRECTORY ${CMAKE_SOURCE_DIR}/results DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

file(READ task00.cpp cpp_source)
# prepare report md file
configure_file(task00.report.md.in.txt ${CMAKE_CURRENT_BINARY_DIR}/task00.report.md @ONLY)
add_custom_command(TARGET ${PROJECT_NAME}
  POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_BINARY_DIR}/task00.report.md" "$<TARGET_FILE_DIR:${PROJECT_NAME}>")