add_executable(surveypropagation SurveyPropagation.cpp)
add_dependencies(apps surveypropagation)
target_link_libraries(surveypropagation PRIVATE Galois::shmem lonestar)
install(TARGETS surveypropagation DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT apps EXCLUDE_FROM_ALL)

# We can get into a situation where these tests do not terminate (issue #119).
add_test_scale(small1 surveypropagation 9 100 300 3 NOT_QUICK)
add_test_scale(small2 surveypropagation 9 100 100 3 NOT_QUICK)

#add_test_scale(web surveypropagation 9 1000000 3000000 3)
