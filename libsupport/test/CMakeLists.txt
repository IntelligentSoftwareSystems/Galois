function(add_test_unit name)
  set(test_name unit-${name})

  add_executable(${test_name} ${name}.cpp)
  target_link_libraries(${test_name} galois_support)

  set(command_line "$<TARGET_FILE:${test_name}>")

  add_test(NAME ${test_name} COMMAND ${command_line})

  # Allow parallel tests
  set_tests_properties(${test_name}
    PROPERTIES
      ENVIRONMENT GALOIS_DO_NOT_BIND_THREADS=1
      LABELS quick
    )
endfunction()

add_test_unit(getenv)
add_test_unit(logging)
