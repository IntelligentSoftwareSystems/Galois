#file(DOWNLOAD http://iss.ices.utexas.edu/projects/galois/downloads/lonestar-cpu-inputs.tar.xz ${CMAKE_CURRENT_BINARY_DIR}/lonestar-cpu-inputs.tar.xz SHOW_PROGRESS)

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/download.cmake "file(DOWNLOAD  http://iss.ices.utexas.edu/projects/galois/downloads/lonestar-cpu-inputs.tar.xz ${CMAKE_CURRENT_BINARY_DIR}/lonestar-cpu-inputs.tar.xz SHOW_PROGRESS)")

add_custom_command(
OUTPUT lonestar-cpu-inputs.tar.xz
COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/download.cmake
)

add_custom_command(
OUTPUT reference/graph_readme.txt
COMMAND ${CMAKE_COMMAND} -E tar xJf lonestar-cpu-inputs.tar.xz && touch reference/graph_readme.txt
DEPENDS lonestar-cpu-inputs.tar.xz
WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/inputs
COMMENT "Unpacking lonestar-cpu-inputs.tar.xz"
VERBATIM
)

add_custom_target(input DEPENDS reference/graph_readme.txt)
