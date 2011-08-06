add_subdirectory(generators)
add_subdirectory(comparisons)
add_subdirectory(graph-convert)

### External Projects

include(ExternalProject)
find_program(WGET wget)

# NB(ddn): bash -c "if [ ! -e ... " is a hack to prevent reexecuting
# commands when not needed
ExternalProject_Add(cats-gens
  PREFIX cats-gens
  # LIST_SEPARATOR is replaced with ";" in generated Makefile
  LIST_SEPARATOR |
  # default is <prefix>/src
  DOWNLOAD_DIR cats-gens/src/cats-gens
  DOWNLOAD_COMMAND bash -c "if [ ! -e makefile ] | then ${WGET} --recursive --no-parent --no-host-directories --cut-dirs=3 --reject \"index.html*\" --quiet http://www.avglab.com/andrew/CATS/gens/ | fi"
  PATCH_COMMAND bash -c "if [ ! -e CMakeLists.txt ] | then patch -p1 < ${CMAKE_CURRENT_SOURCE_DIR}/cats-gens.patch | fi"
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}
)

find_program(UNZIP unzip)
# Manually specify steps because ExternalProject doesn't know about unzip
ExternalProject_Add(triangle
  PREFIX triangle
  LIST_SEPARATOR |
  DOWNLOAD_DIR triangle/src/triangle
  DOWNLOAD_COMMAND bash -c "if [ ! -e triangle.zip ] | then ${WGET} --quiet http://www.netlib.org/voronoi/triangle.zip | fi"
  UPDATE_COMMAND bash -c "if [ ! -e makefile ] | then ${UNZIP} triangle.zip | fi"
  PATCH_COMMAND bash -c "if [ ! -e CMakeLists.txt ] | then patch -p1 < ${CMAKE_CURRENT_SOURCE_DIR}/triangle.patch | fi"
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}
)

find_program(TAR tar)
ExternalProject_Add(hi_pr
  PREFIX hi_pr
  LIST_SEPARATOR |
  DOWNLOAD_DIR hi_pr/src/hi_pr
  URL http://www.avglab.com/andrew/soft/hipr.tar
  PATCH_COMMAND bash -c "if [ ! -e CMakeLists.txt ] | then patch -p1 < ${CMAKE_CURRENT_SOURCE_DIR}/hi_pr.patch | fi"
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}
)

add_custom_target(more-tools)
set_target_properties(hi_pr triangle cats-gens PROPERTIES EXCLUDE_FROM_ALL true)
add_dependencies(more-tools hi_pr triangle cats-gens)
