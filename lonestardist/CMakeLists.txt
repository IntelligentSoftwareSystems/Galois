# NOTE: this is required because 
# target_link_libraries() does not work with cuda_add_library() 
# (in old versions of CMake)
include_directories(${PROJECT_SOURCE_DIR}/libgluon/include)
include_directories(${PROJECT_SOURCE_DIR}/lonestardist/include)

add_library(distbench STATIC src/DistBenchStart.cpp src/DistributedGraphLoader.cpp)
target_include_directories(distbench PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(distbench Galois::cusp Galois::gluon)

if(EXISTS /net/ohm/export/iss/dist-outputs)
  set(BASEOUTPUT /net/ohm/export/iss/dist-outputs)
elseif(EXISTS /net/ohm/export/cdgc/dist-outputs)
  set(BASEOUTPUT /net/ohm/export/cdgc/dist-outputs)
elseif(EXISTS /workspace/dist-outputs)
  set(BASEOUTPUT /workspace/dist-outputs)
endif()

if(ENABLE_HETERO_GALOIS)
  # turn on cuda for distbench as well
  target_compile_definitions(distbench PRIVATE __GALOIS_HET_CUDA__=1)

  # for debugging
  add_definitions(-D__GALOIS_CUDA_CHECK_ERROR__)
  if(CMAKE_BUILD_TYPE MATCHES "Debug")
    add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>")

    add_definitions(-D__GALOIS_DEBUG_WORKLIST__)
  endif()
endif()

# CuSP uses DGAccumulator which uses LCI if enabled
if (USE_LCI)
  target_include_directories(distbench PUBLIC ${LWCI_INCLUDE})
endif()

function(app_dist name)
  set(options NO_GPU)
  set(one_value_args)
  set(multi_value_args)
  cmake_parse_arguments(X "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  FILE(GLOB CPPSOURCES ${name}*.cpp)
  add_executable(${name} ${CPPSOURCES})
  add_dependencies(apps ${name})
  target_link_libraries(${name} lonestar Galois::shmem LLVMSupport)
  install(TARGETS ${name} DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT apps EXCLUDE_FROM_ALL)

  target_link_libraries(${name} distbench)
  if(REPORT_PER_ROUND_STATS)
    target_compile_definitions(${name} PRIVATE DIST_PER_ROUND_TIMER=1)
  endif()
  if(REPORT_COMM_STATS)
    target_compile_definitions(${name} PRIVATE MORE_COMM_STATS=1)
  endif()
  if(USE_BARE_MPI)
    target_compile_definitions(${name} PRIVATE __GALOIS_BARE_MPI_COMMUNICATION__=1)
  endif()

  if(ENABLE_HETERO_GALOIS AND NOT ${X_NO_GPU})
    target_compile_definitions(${name} PRIVATE __GALOIS_HET_CUDA__=1)
    target_link_libraries(${name} ${name}_cuda)

    FILE(GLOB CUSOURCES ${name}*.cu)
    add_library(${name}_cuda ${CUSOURCES})
    target_link_libraries(${name}_cuda Galois::gpu)
    set_property(TARGET ${name}_cuda PROPERTY CUDA_STANDARD 14)
  endif()
endfunction()

set(RESULT_CHECKER ${PROJECT_SOURCE_DIR}/scripts/result_checker.py)
cmake_host_system_information(RESULT HOSTNAME QUERY HOSTNAME)

function(add_test_dist_and_verify app input type part N np)
  set(options GPU NOT_QUICK)
  set(one_value_args)
  set(multi_value_args)
  cmake_parse_arguments(X "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  math(EXPR t "(${N} / ${np})")
  string(REPLACE "_" ";" app_list ${app})
  list(GET app_list 0 app_id)
  set(OUTPUT ${BASEOUTPUT}/${input}.${app_id})

  if (EXISTS ${OUTPUT})
    add_test(run-${app}-${type}-${input}-${part}-${np} mpiexec --bind-to none -n ${np} --env GALOIS_DO_NOT_BIND_THREADS 1 ./${app} ${X_UNPARSED_ARGUMENTS} -t=${t} -partition=${part} -verify)
    add_test(verify-${app}-${type}-${input}-${part}-${np} python ${RESULT_CHECKER} -t=0.01 -sort=1 -delete=1 ${OUTPUT} output_${HOSTNAME}_*.log)
  else()
    add_test(run-${app}-${type}-${input}-${part}-${np} mpiexec --bind-to none -n ${np} --env GALOIS_DO_NOT_BIND_THREADS 1 ./${app} ${X_UNPARSED_ARGUMENTS} -t=${t} -partition=${part})
  endif()

  if (${X_GPU})
    set_tests_properties(run-${app}-${type}-${input}-${part}-${np} PROPERTIES RUN_SERIAL true)
  endif()

  # Don't run dist tests in CI since they are broken in too many cases right now.
  #if (NOT ${X_NOT_QUICK})
  #  set_tests_properties(run-${app}-${type}-${input}-${part}-${np}
  #    PROPERTIES ENVIRONMENT GALOIS_DO_NOT_BIND_THREADS=1 LABELS quick)
  #endif()
endfunction()

function(add_test_dist_for_partitions app input type num_threads num_gpus part)
  set(partitions ${num_threads})
  set(thr ${num_threads})
  while (${thr} GREATER 1)
    math(EXPR thr "${thr} / 2")
    list(APPEND partitions ${thr})
  endwhile()
  list(REVERSE partitions)

  foreach(np ${partitions})
    if (np GREATER 1)
      add_test_dist_and_verify(${app} ${input} ${type}-cpu ${part} ${num_threads} 1 ${ARGN})
    endif()
    add_test_dist_and_verify(${app} ${input} ${type}-cpu ${part} ${num_threads} ${np} ${ARGN})
  endforeach()

  if (NOT ENABLE_HETERO_GALOIS)
    return()
  endif()

  if (num_gpus LESS_EQUAL 0)
    return()
  endif()

  if(num_gpus GREATER_EQUAL num_threads)
    message(FATAL_ERROR "number of test gpus (${num_gpus}) should be less than number of test threads (${num_threads})")
  endif()

  set(PSET "-pset=")
  foreach(np RANGE 1 ${num_gpus})
    set(PSET "${PSET}g")
    add_test_dist_and_verify(${app} ${input} ${type}-gpu ${part} ${num_threads} ${np} GPU ${ARGN} -num_nodes=1 ${PSET})
  endforeach(np)
  set(PSET "${PSET}c")
  math(EXPR np "(${G} + 1)")
  add_test_dist_and_verify(${app} ${input} ${type}-cpugpu ${part} ${num_threads} ${np} GPU ${ARGN} -num_nodes=1 ${PSET} -scalegpu=3)
endfunction()

function(add_test_dist app input)
  set(options NO_GPU NO_ASYNC)
  set(one_value_args)
  set(multi_value_args)
  cmake_parse_arguments(X "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  set(num_gpus ${NUM_TEST_GPUS})
  if (${X_NO_GPU})
    set(num_gpus 0)
  endif()

  set(num_threads ${NUM_TEST_THREADS})

  foreach (part oec iec cvc cvc-iec hovc hivc)
    if (NOT ${X_NO_ASYNC})
      add_test_dist_for_partitions(${app} ${input} sync ${num_threads} ${num_gpus} ${part} ${X_UNPARSED_ARGUMENTS} -exec=Sync)
      add_test_dist_for_partitions(${app} ${input} async ${num_threads} ${num_gpus} ${part} ${X_UNPARSED_ARGUMENTS} -exec=Async)
    else()
      add_test_dist_for_partitions(${app} ${input} sync ${num_threads} ${num_gpus} ${part} ${X_UNPARSED_ARGUMENTS})
    endif()
  endforeach()
endfunction()
add_subdirectory(bc)
add_subdirectory(bfs)
add_subdirectory(cc)
add_subdirectory(kcore)
add_subdirectory(pagerank)
add_subdirectory(sgd)
add_subdirectory(sssp)
#add_subdirectory(kclique)

add_subdirectory(partition)

# currently tc only has a GPU implementation
if(ENABLE_HETERO_GALOIS)
  # tc does not currently build
  add_subdirectory(tc EXCLUDE_FROM_ALL)
endif()
