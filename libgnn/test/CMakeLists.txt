add_executable(gnngraph-test gnngraph-test.cpp)
target_link_libraries(gnngraph-test galois_gnn)
add_test(NAME gnngraph-test COMMAND gnngraph-test)

if (NOT GALOIS_ENABLE_GPU)
  add_executable(convlayer-test convlayer-test.cpp)
  target_link_libraries(convlayer-test galois_gnn)
  add_test(NAME convlayer-test COMMAND convlayer-test)

  add_executable(sage-layer-test sage-layer-test.cpp)
  target_link_libraries(sage-layer-test galois_gnn)
  add_test(NAME sage-layer-test COMMAND sage-layer-test)

  add_executable(l2norm-layer-test l2norm-layer-test.cpp)
  target_link_libraries(l2norm-layer-test galois_gnn)
  add_test(NAME l2norm-layer-test COMMAND l2norm-layer-test)

  add_executable(softmaxlayer-test softmaxlayer-test.cpp)
  target_link_libraries(softmaxlayer-test galois_gnn)
  add_test(NAME softmaxlayer-test COMMAND softmaxlayer-test)
  
  add_executable(sigmoidlayer-test sigmoidlayer-test.cpp)
  target_link_libraries(sigmoidlayer-test galois_gnn)
  add_test(NAME sigmoidlayer-test COMMAND sigmoidlayer-test)

  add_executable(gnnconstruct-test gnnconstruct-test.cpp)
  target_link_libraries(gnnconstruct-test galois_gnn)
  add_test(NAME gnnconstruct-test COMMAND gnnconstruct-test)
  
  add_executable(gnnfb-test gnnfb-test.cpp)
  target_link_libraries(gnnfb-test galois_gnn)
  add_test(NAME gnnfb-test COMMAND gnnfb-test)
  
  add_executable(adam-test adam-test.cpp)
  target_link_libraries(adam-test galois_gnn)
  add_test(NAME adam-test COMMAND adam-test)
  
  add_executable(accuracy-test accuracy-test.cpp)
  target_link_libraries(accuracy-test galois_gnn)
  add_test(NAME accuracy-test COMMAND accuracy-test)
  
  add_executable(epoch-test epoch-test.cpp)
  target_link_libraries(epoch-test galois_gnn)
  add_test(NAME epoch-test COMMAND epoch-test)

  add_executable(multilabel-epoch-test multilabel-epoch-test.cpp)
  target_link_libraries(multilabel-epoch-test galois_gnn)
  add_test(NAME multilabel-epoch-test COMMAND multilabel-epoch-test)

  # TODO figure out how to make this test run in parallel
  add_executable(aggregate-sync-test aggregate-sync-test.cpp)
  target_link_libraries(aggregate-sync-test galois_gnn)
  #add_test(NAME aggregate-sync-test COMMAND GALOIS_DO_NOT_BIND_THREADS=1 mpirun -n=4 ./aggregate-sync-test)

  add_executable(weight-sync-test weight-sync-test.cpp)
  target_link_libraries(weight-sync-test galois_gnn)

  add_executable(multilabel-read multilabel-read.cpp)
  target_link_libraries(multilabel-read galois_gnn)
  add_test(NAME multilabel-read COMMAND multilabel-read)

  add_executable(f1-test f1-test.cpp)
  target_link_libraries(f1-test galois_gnn)
  add_test(NAME f1-test COMMAND f1-test)

  add_executable(sample-test sample-test.cpp)
  target_link_libraries(sample-test galois_gnn)
  add_test(NAME sample-test COMMAND sample-test)
else()
  add_executable(gpu-convlayer-test gpu-convlayer-test.cpp)
  target_link_libraries(gpu-convlayer-test galois_gnn)
  add_test(NAME gpu-convlayer-test COMMAND gpu-convlayer-test)

  add_executable(gpu-softmaxlayer-test gpu-softmaxlayer-test.cpp)
  target_link_libraries(gpu-softmaxlayer-test galois_gnn)
  add_test(NAME gpu-softmaxlayer-test COMMAND gpu-softmaxlayer-test)

  add_executable(gpu-adam-test gpu-adam-test.cpp)
  target_link_libraries(gpu-adam-test galois_gnn)
  add_test(NAME gpu-adam-test COMMAND gpu-adam-test)

  add_executable(gpu-epoch-test gpu-epoch-test.cpp)
  target_link_libraries(gpu-epoch-test galois_gnn)
  #add_test(NAME gpu-epoch-test COMMAND gpu-epoch-test)

  add_executable(gpu-aggregate-sync-test gpu-aggregate-sync-test.cpp)
  target_link_libraries(gpu-aggregate-sync-test galois_gnn)
endif()

# TODO multi host tests?
