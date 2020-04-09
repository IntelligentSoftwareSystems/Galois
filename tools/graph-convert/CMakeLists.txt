app(graph-convert graph-convert.cpp)
install(TARGETS graph-convert EXPORT GaloisTargets RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT bin)

app(graph-convert-huge graph-convert-huge.cpp)
install(TARGETS graph-convert-huge EXPORT GaloisTargets RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT bin)

target_link_libraries(graph-convert-huge ${Boost_IOSTREAMS_LIBRARY})
