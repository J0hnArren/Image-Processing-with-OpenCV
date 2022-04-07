project(lab04)
add_executable(lab04 json.hpp lab04.cpp lab04.report.md.in.txt)
target_link_libraries(lab04 ${OpenCV_LIBS})
install(DIRECTORY ${CMAKE_SOURCE_DIR}/data DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

file(READ lab04.cpp cpp_source)
# prepare report md file
configure_file(lab04.report.md.in.txt ${CMAKE_CURRENT_BINARY_DIR}/lab04.report.md @ONLY)
add_custom_command(TARGET ${PROJECT_NAME}
  POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_BINARY_DIR}/lab04.report.md" "$<TARGET_FILE_DIR:${PROJECT_NAME}>")