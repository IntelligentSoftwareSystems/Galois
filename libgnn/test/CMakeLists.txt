add_executable(gnngraph-test gnngraph-test.cpp)
target_link_libraries(gnngraph-test galois_gnn)
add_test(NAME gnngraph-test COMMAND gnngraph-test)

add_executable(convlayer-test convlayer-test.cpp)
target_link_libraries(convlayer-test galois_gnn)
add_test(NAME convlayer-test COMMAND convlayer-test)

add_executable(softmaxlayer-test softmaxlayer-test.cpp)
target_link_libraries(softmaxlayer-test galois_gnn)
add_test(NAME softmaxlayer-test COMMAND softmaxlayer-test)

add_executable(gnnconstruct-test gnnconstruct-test.cpp)
target_link_libraries(gnnconstruct-test galois_gnn)
add_test(NAME gnnconstruct-test COMMAND gnnconstruct-test)

add_executable(gnnfb-test gnnfb-test.cpp)
target_link_libraries(gnnfb-test galois_gnn)
add_test(NAME gnnfb-test COMMAND gnnfb-test)

add_executable(adam-test adam-test.cpp)
target_link_libraries(adam-test galois_gnn)
add_test(NAME adam-test COMMAND adam-test)

# TODO multi host tests?
