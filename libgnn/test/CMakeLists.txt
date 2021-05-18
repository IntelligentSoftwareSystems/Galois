add_executable(mkl_micro mkl_micro.cpp)
target_link_directories(mkl_micro PUBLIC ${MKL_LIBRARIES})
target_include_directories(mkl_micro PUBLIC
  ${MKL_INCLUDE_DIRS}
)
target_link_libraries(mkl_micro ${INTEL_LIBS})

add_executable(mkl_micro_sgalois mkl_micro.cpp)
target_link_libraries(mkl_micro_sgalois galois_gnn)
target_compile_definitions(mkl_micro_sgalois PUBLIC USE_SHARED_GALOIS=1)

add_executable(mkl_micro_dgalois mkl_micro.cpp)
target_link_libraries(mkl_micro_dgalois galois_gnn)
target_compile_definitions(mkl_micro_dgalois PUBLIC USE_DIST_GALOIS=1)

################################################################################

add_executable(gnngraph-test gnngraph-test.cpp)
target_link_libraries(gnngraph-test galois_gnn)
add_test(NAME gnngraph-test COMMAND gnngraph-test)

# multihost testing things
set(hosts)
set(host 12)
while (${host} GREATER 1)
  list(APPEND hosts ${host})
  math(EXPR host "${host} - 1")
endwhile()
list(APPEND hosts "1")

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

  add_executable(aggregate-sync-test aggregate-sync-test.cpp)
  target_link_libraries(aggregate-sync-test galois_gnn)
  foreach(host_count ${hosts})
    add_test(NAME run-aggsync-${host_count} COMMAND mpiexec --bind-to none -n ${host_count} ./aggregate-sync-test)
    set_tests_properties(run-aggsync-${host_count} PROPERTIES ENVIRONMENT "GALOIS_DO_NOT_BIND_THREADS=1")
  endforeach()

  add_executable(back-conv-test back-conv-test.cpp)
  target_link_libraries(back-conv-test galois_gnn)
  foreach(host_count ${hosts})
    add_test(NAME run-back-conv-${host_count} COMMAND mpiexec --bind-to none -n ${host_count} ./back-conv-test)
    set_tests_properties(run-back-conv-${host_count} PROPERTIES ENVIRONMENT "GALOIS_DO_NOT_BIND_THREADS=1")
  endforeach()

  add_executable(multilabel-read multilabel-read.cpp)
  target_link_libraries(multilabel-read galois_gnn)
  add_test(NAME multilabel-read COMMAND multilabel-read)

  add_executable(f1-test f1-test.cpp)
  target_link_libraries(f1-test galois_gnn)
  add_test(NAME f1-test COMMAND f1-test)

  add_executable(sample-test sample-test.cpp)
  target_link_libraries(sample-test galois_gnn)
  add_test(NAME sample-test COMMAND sample-test)

  add_executable(sample-bit-test sample-bit-test.cpp)
  target_link_libraries(sample-bit-test galois_gnn)
  add_test(NAME sample-bit-test COMMAND sample-bit-test)
else()
  add_executable(gpu-sage-layer-test gpu-sage-layer-test.cpp)
  target_link_libraries(gpu-sage-layer-test galois_gnn)
  add_test(NAME gpu-sage-layer-test COMMAND gpu-sage-layer-test)

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

  set(gpu_hosts)
  set(gpu_host 3) #TODO(lhc) more than 4 gpus, test failed
                  #          seems like it happened due to graph size 0.
                  #          so let me postpone this
  while (${gpu_host} GREATER 1)
    list(APPEND gpu_hosts ${gpu_host})
    math(EXPR gpu_host "${gpu_host} - 1")
  endwhile()
  list(APPEND gpu_hosts "1")

  add_executable(gpu-back-conv-test gpu-back-conv-test.cpp)
  target_link_libraries(gpu-back-conv-test galois_gnn)
  foreach(gpu_host_count ${gpu_hosts})
    set(PSET "-pset=")
    foreach(iter RANGE 1 ${gpu_host_count})
      set(PSET "${PSET}g")
    endforeach()
    add_test(NAME run-gpu-back-conv-${gpu_host_count} COMMAND mpiexec --bind-to none -n ${gpu_host_count} ./gpu-back-conv-test ${PSET} -numNodes=1)
    set_tests_properties(run-gpu-back-conv-${gpu_host_count} PROPERTIES ENVIRONMENT "GALOIS_DO_NOT_BIND_THREADS=1")
  endforeach()
endif()

# TODO multi host tests?
