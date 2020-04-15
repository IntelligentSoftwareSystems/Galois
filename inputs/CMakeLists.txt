file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/download.cmake "file(DOWNLOAD http://iss.oden.utexas.edu/projects/galois/downloads/small_inputs_for_lonestar_test.tar.gz ${CMAKE_CURRENT_BINARY_DIR}/lonestar-cpu-inputs.tar.gz SHOW_PROGRESS)")

add_custom_command(
OUTPUT lonestar-cpu-inputs.tar.gz
COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/download.cmake
)

add_custom_command(
OUTPUT small_inputs
COMMAND ${CMAKE_COMMAND} -E tar xJf lonestar-cpu-inputs.tar.gz
DEPENDS lonestar-cpu-inputs.tar.gz
WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/inputs
COMMENT "Unpacking lonestar-cpu-inputs.tar.gz"
VERBATIM
)

add_custom_target(input DEPENDS small_inputs)
