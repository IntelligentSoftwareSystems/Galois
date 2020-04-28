add_executable(louvainClustering louvainClustering.cpp)
add_dependencies(apps louvainClustering)
target_link_libraries(louvainClustering PRIVATE Galois::shmem lonestar)
install(TARGETS louvainClustering DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT apps EXCLUDE_FROM_ALL)
