add_executable(graph-convert-standalone ../graph-convert/graph-convert.cpp)
target_link_libraries(graph-convert-standalone galois-nothreads)
install(TARGETS graph-convert-standalone EXPORT GaloisTargets RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT bin)
