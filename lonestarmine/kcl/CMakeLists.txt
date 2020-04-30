add_executable(kcl kcl.cpp)
add_dependencies(apps kcl)
target_link_libraries(kcl PRIVATE Galois::pangolin LLVMSupport)
install(TARGETS kcl DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT apps EXCLUDE_FROM_ALL)

add_test_mine(small1 kcl "${BASEINPUT}/Mining/citeseer.csgr")

if (ENABLE_HETERO_GALOIS)
  add_executable(kcl_gpu kcl_gpu.cpp kcl.cu)
  add_dependencies(apps kcl_gpu)
  target_link_libraries(kcl_gpu PRIVATE Galois::pangolin_gpu LLVMSupport)
  target_compile_definitions(kcl_gpu PRIVATE __GALOIS_HET_CUDA__=1)
  install(TARGETS kcl_gpu DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT apps EXCLUDE_FROM_ALL)
endif()
