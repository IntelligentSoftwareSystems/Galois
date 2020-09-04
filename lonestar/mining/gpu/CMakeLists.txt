include_directories(include)

function(add_test_mine type app)
  set(options NOT_QUICK)
  set(one_value_args)
  set(multi_value_args REQUIRES COMMAND_PREFIX)
  cmake_parse_arguments(X "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  set(threads)
  set(thr "${GALOIS_NUM_TEST_THREADS}")
  while (${thr} GREATER 1)
    list(APPEND threads ${thr})
    math(EXPR thr "${thr} / 2")
  endwhile()
  list(APPEND threads "1")

  foreach (thr ${threads})
    set(name run-${type}-${app}-${thr})
    add_test(NAME ${name} COMMAND ${app} ${X_UNPARSED_ARGUMENTS} -t ${thr})
    if (NOT ${X_NOT_QUICK})
      # Allow parallel tests
      set_tests_properties(${name}
        PROPERTIES ENVIRONMENT GALOIS_DO_NOT_BIND_THREADS=1 LABELS quick)
    endif()
  endforeach()
endfunction(add_test_mine)

add_subdirectory(frequent-subgraph-mining)
add_subdirectory(k-clique-listing)
add_subdirectory(motif-counting)
add_subdirectory(triangle-counting)
