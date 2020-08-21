function(add_test_unit name)
  set(options)
  set(multi_value_args REQUIRES COMMAND_PREFIX)
  cmake_parse_arguments(X "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  foreach(required ${X_REQUIRES})
    if(${${required}} MATCHES "TRUE")
    else()
      message(STATUS "NOT compiling ${name} (missing: ${required})")
      return()
    endif()
  endforeach()

  set(test_name unit-${name})

  add_executable(${test_name} ${name}.cpp)
  target_link_libraries(${test_name} galois_shmem lonestar)

  set(commandline ${X_COMMAND_PREFIX})
  list(APPEND commandline "$<TARGET_FILE:${test_name}>")
  list(APPEND commandline ${X_UNPARSED_ARGUMENTS})

  add_test(NAME ${test_name} COMMAND ${commandline})

  # Allow parallel tests
  set_tests_properties(${test_name}
    PROPERTIES
      ENVIRONMENT GALOIS_DO_NOT_BIND_THREADS=1
      LABELS quick
    )
endfunction()

add_test_unit(acquire)
add_test_unit(bandwidth)
add_test_unit(barriers 1024 2)
add_test_unit(empty-member-lcgraph)
add_test_unit(flatmap)
add_test_unit(floatingPointErrors)
add_test_unit(foreach)
add_test_unit(forward-declare-graph)
add_test_unit(gcollections)
add_test_unit(graph)
add_test_unit(graph-compile)
add_test_unit(gslist)
add_test_unit(hwtopo)
add_test_unit(lc-adaptor)
add_test_unit(lock)
add_test_unit(loop-overhead REQUIRES OPENMP_FOUND)
add_test_unit(mem)
add_test_unit(morphgraph)
add_test_unit(move)
add_test_unit(oneach)
add_test_unit(papi 2)
add_test_unit(pc)
add_test_unit(prefixsum)
add_test_unit(reduction)
add_test_unit(sort)
add_test_unit(static)
add_test_unit(traits)
add_test_unit(twoleveliteratora)
add_test_unit(wakeup-overhead)
add_test_unit(worklists-compile)
add_test_unit(morphgraph-removal)
add_test_unit(wfl)
