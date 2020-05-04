function(app_gpu name)
  set(options NO_GPU)
  set(one_value_args)
  set(multi_value_args)
  cmake_parse_arguments(X "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  add_executable(${name}_gpu ${name}.cu)
  install(TARGETS ${name}_gpu DESTINATION "${CMAKE_INSTALL_BINDIR}" EXCLUDE_FROM_ALL)

  if(ENABLE_HETERO_GALOIS AND NOT ${X_NO_GPU})

    #target_compile_options(${name}_gpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-dc>)

    target_link_libraries(${name}_gpu Galois::gpu)
    set_property(TARGET ${name}_gpu PROPERTY CUDA_STANDARD 14)
    set_property(TARGET ${name}_gpu PROPERTY CUDA_SEPARABLE_COMPILATION ON)
  endif()
endfunction()

add_subdirectory(bh)

