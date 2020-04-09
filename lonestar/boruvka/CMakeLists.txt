add_executable(boruvka Boruvka.cpp)
add_dependencies(apps boruvka)
target_link_libraries(boruvka PRIVATE galois_shmem lonestar)
install(TARGETS boruvka DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT apps EXCLUDE_FROM_ALL)

add_test_scale(small1 boruvka "${BASEINPUT}/scalefree/rmat10.gr")
add_test_scale(small2 boruvka "${BASEINPUT}/reference/structured/rome99.gr")
#add_test_scale(web boruvka "${BASEINPUT}/road/USA-road-d.USA.gr")
