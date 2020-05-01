add_library(lonestar STATIC src/BoilerPlate.cpp)

target_include_directories(lonestar PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(lonestar Galois::shmem LLVMSupport)

function(add_test_scale type app)
  set(options NOT_QUICK)
  set(one_value_args)
  set(multi_value_args REQUIRES COMMAND_PREFIX)
  cmake_parse_arguments(X "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  set(threads)
  set(thr "${NUM_TEST_THREADS}")
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
endfunction(add_test_scale)

add_subdirectory(aigRewriting)
add_subdirectory(barneshut)
add_subdirectory(betweennesscentrality) 
add_subdirectory(bfs)
add_subdirectory(bipart)
add_subdirectory(boruvka)
add_subdirectory(connectedcomponents)
add_subdirectory(delaunayrefinement)
add_subdirectory(delaunaytriangulation)
add_subdirectory(gmetis)
add_subdirectory(independentset)
add_subdirectory(kcore)
add_subdirectory(matching)
add_subdirectory(matrixcompletion)
add_subdirectory(pagerank)
add_subdirectory(pointstoanalysis)
add_subdirectory(preflowpush)
add_subdirectory(SPRoute)
add_subdirectory(sssp)
add_subdirectory(triangles)
add_subdirectory(tutorial_examples)
add_subdirectory(ktruss)
add_subdirectory(clustering)

