function(add_test_unit name)
  set(test_name unit-${name})

  add_executable(${test_name} ${name}.cpp)
  target_link_libraries(${test_name} Galois::wmd Galois::cusp Galois::dist_async Galois::gluon)

  set(command_line "$<TARGET_FILE:${test_name}>")

  add_test(NAME ${test_name} COMMAND ${command_line} ${Galois_SOURCE_DIR}/inputs/wmd/data.00001.csv)
endfunction()

add_test_unit(wmd-graph-build)
