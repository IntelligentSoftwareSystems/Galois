find_package(OpenMP)

set(hosts)
set(host 12)
while (${host} GREATER 1)
  list(APPEND hosts ${host})
  math(EXPR host "${host} - 1")
endwhile()
list(APPEND hosts "1")

add_executable(gnngraph-test gnngraph-test.cpp)
target_link_libraries(gnngraph-test galois_gnn)
add_test(NAME gnngraph-test COMMAND gnngraph-test)

if (NOT GALOIS_ENABLE_GPU)
  set(GALOIS_TESTS
      ${GALOIS_TESTS}
      convlayer-test
      sage-layer-test
      # l2norm-layer-test TODO(hc): L2Norm is not maintained
      softmaxlayer-test
      sigmoidlayer-test
      gnnconstruct-test
      gnnfb-test
      adam-test
      accuracy-test
      epoch-test
      multilabel-epoch-test
      multilabel-read
      # f1-test TODO(hc): it is a multilable test and will be fixed later
      sample-bit-test
      gcn-sample-edge-test
  )
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
else()
  set(GALOIS_TESTS
      ${GALOIS_TESTS}
      gpu-sage-layer-test
      gpu-convlayer-test
      gpu-softmaxlayer-test
      gpu-adam-test
      gpu-epoch-test
     )
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

message("Galois Tests..")
foreach(galois_test ${GALOIS_TESTS})
  add_executable(${galois_test} ${galois_test}.cpp)
  target_link_libraries(${galois_test} galois_gnn)
  add_test(NAME ${galois_test} COMMAND ${galois_test})
endforeach()

add_executable(remapverify remapverify.cpp)
target_link_libraries(remapverify galois_gnn)
target_compile_definitions(remapverify PUBLIC USE_DIST_GALOIS=1)

# MKL Test
set(MKL_TESTS
    mkl_micro_sgalois
    mkl_micro_dgalois
    mkl_micro_delete_galois
    single_mkl_micro_sgalois
    single_mkl_micro_dgalois
    mkl_micro
    mkl_micro_omp
    single_mkl_micro_omp
)

add_executable(mkl_micro_sgalois mkl_micro.cpp)
target_link_libraries(mkl_micro_sgalois PUBLIC galois_gnn)
target_compile_definitions(mkl_micro_sgalois PUBLIC USE_SHARED_GALOIS=1)

add_executable(mkl_micro_dgalois mkl_micro.cpp)
target_link_libraries(mkl_micro_dgalois PUBLIC galois_gnn)
target_compile_definitions(mkl_micro_dgalois PUBLIC USE_DIST_GALOIS=1)

add_executable(mkl_micro_delete_galois mkl_micro.cpp)
target_link_libraries(mkl_micro_delete_galois PUBLIC galois_gnn)
target_compile_definitions(mkl_micro_delete_galois PUBLIC USE_SHARED_GALOIS_DELETE=1)

add_executable(single_mkl_micro_sgalois single_mkl_micro.cpp)
target_link_libraries(single_mkl_micro_sgalois PUBLIC galois_gnn_single)
target_compile_definitions(single_mkl_micro_sgalois PUBLIC USE_SHARED_GALOIS=1)

add_executable(single_mkl_micro_dgalois single_mkl_micro.cpp)
target_link_libraries(single_mkl_micro_dgalois PUBLIC galois_gnn_single)
target_compile_definitions(single_mkl_micro_dgalois PUBLIC USE_DIST_GALOIS=1)

add_executable(mkl_micro mkl_micro.cpp)

add_executable(mkl_micro_omp mkl_micro.cpp)
target_link_libraries(mkl_micro_omp PUBLIC ${INTEL_LIBS} OpenMP::OpenMP_CXX)
target_compile_definitions(mkl_micro_omp PUBLIC USE_OMP=1)

add_executable(single_mkl_micro_omp single_mkl_micro.cpp)
target_link_libraries(single_mkl_micro_omp PUBLIC ${SINGLE_INTEL_LIBS} OpenMP::OpenMP_CXX)
target_compile_definitions(single_mkl_micro_omp PUBLIC USE_OMP=1)

foreach(mkl_test ${MKL_TESTS})
  target_compile_options(${mkl_test} PUBLIC
      $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
  target_include_directories(${mkl_test} PUBLIC
      $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
  target_link_libraries(${mkl_test} PUBLIC $<LINK_ONLY:MKL::MKL>)
endforeach()

add_executable(gstl_test gstl_test.cpp)
target_link_libraries(gstl_test galois_shmem)
