add_executable(ktruss K-Truss.cpp)
add_dependencies(apps ktruss)
target_link_libraries(ktruss PRIVATE Galois::shmem lonestar)
install(TARGETS ktruss DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT apps EXCLUDE_FROM_ALL)
add_custom_command(
  TARGET ktruss
  POST_BUILD
  COMMAND ln -sf ${CMAKE_CURRENT_SOURCE_DIR}/bmktest2.py bmktest2.py
)

add_executable(verifyKTruss Verify.cpp)
add_dependencies(apps verifyKTruss)
target_link_libraries(verifyKTruss PRIVATE Galois::shmem lonestar)
install(TARGETS verifyKTruss DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT apps EXCLUDE_FROM_ALL)
