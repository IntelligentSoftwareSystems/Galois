file(GLOB Sources 
  subjectgraph/aig/*.cpp
  algorithms/*.cpp
  parsers/*.cpp
  writers/*.cpp
  misc/util/*.cpp
  functional/*.cpp
  xxHash/xxhash.c
)

add_executable(aig-rewriting-cpu main.cpp ${Sources})
add_dependencies(apps aig-rewriting-cpu)
target_link_libraries(aig-rewriting-cpu PRIVATE Galois::shmem lonestar)
install(TARGETS aig-rewriting-cpu DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT apps EXCLUDE_FROM_ALL)

target_include_directories(aig-rewriting-cpu PRIVATE
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/subjectgraph/aig>"
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/algorithms>"
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/parsers>"
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/writers>"
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/misc>"
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/misc/util>"
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/functional>"
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/xxHash>"
)

add_test_scale(small1 aig-rewriting-cpu -AIG "${BASEINPUT}/eda/logic-synthesis/EPFL/arithmetic/adder/aiger/adder.aig" -v)
add_test_scale(small2 aig-rewriting-cpu -AIG "${BASEINPUT}/eda/logic-synthesis/EPFL/random_control/voter/aiger/voter.aig" -v)
