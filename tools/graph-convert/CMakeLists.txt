app(graph-convert graph-convert.cpp)
install(TARGETS graph-convert EXPORT GaloisTargets RUNTIME DESTINATION "${INSTALL_BIN_DIR}" COMPONENT bin)

app(graph-convert-huge graph-convert-huge.cpp)
install(TARGETS graph-convert-huge EXPORT GaloisTargets RUNTIME DESTINATION "${INSTALL_BIN_DIR}" COMPONENT bin)

target_include_directories(graph-convert-huge PUBLIC 
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/libdist/include>)
target_link_libraries(graph-convert-huge z boost_iostreams)
