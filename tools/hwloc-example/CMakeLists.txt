file(GLOB Sources *.c)
add_executable(hwloc-galois-example ${Sources})
target_link_libraries(hwloc-galois-example hwloc)

