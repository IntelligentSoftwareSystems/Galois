add_executable(sgl sgl.cpp)
add_dependencies(apps sgl)
target_link_libraries(sgl PRIVATE Galois::pangolin LLVMSupport)
install(TARGETS sgl DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT apps EXCLUDE_FROM_ALL)
