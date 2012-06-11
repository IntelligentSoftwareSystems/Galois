include_directories("include")
set(input ${CMAKE_CURRENT_BINARY_DIR}/inputs)
set(input_src ${CMAKE_CURRENT_SOURCE_DIR}/inputs)

add_subdirectory(inputs)
add_subdirectory(apps)
