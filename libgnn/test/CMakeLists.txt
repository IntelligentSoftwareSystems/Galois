add_executable(gnngraph-test gnngraph-test.cpp)
target_link_libraries(gnngraph-test galois_gnn)
add_test(NAME gnngraph-test COMMAND gnngraph-test)

add_executable(convlayer-test convlayer-test.cpp)
target_link_libraries(convlayer-test galois_gnn)
add_test(NAME convlayer-test COMMAND convlayer-test)

# TODO multi host tests
