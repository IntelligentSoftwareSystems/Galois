add_executable(graph-convert graph-convert.cpp)
target_link_libraries(graph-convert galois_shmem LLVMSupport)
install(TARGETS graph-convert
  EXPORT GaloisTargets
  DESTINATION "${CMAKE_INSTALL_BINDIR}"
  COMPONENT tools
)

add_executable(graph-convert-huge graph-convert-huge.cpp)
target_link_libraries(graph-convert-huge galois_shmem LLVMSupport)
if (TARGET Boost::Boost)
  target_link_libraries(graph-convert-huge Boost::Boost)
else()
  target_link_libraries(graph-convert-huge Boost::iostreams)
endif()
install(TARGETS graph-convert-huge
  EXPORT GaloisTargets
  DESTINATION "${CMAKE_INSTALL_BINDIR}"
  COMPONENT tools
)
