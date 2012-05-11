add_custom_target(more-suites-inputs-pbbs)
add_dependencies(more-suites-inputs more-suites-inputs-pbbs)

function(executable name)
  add_executable(${name} ${ARGN})
  target_link_libraries(${name} pbbs ${extra_libs})
  addsuite(${name})
endfunction()

add_subdirectory(geometryData)
add_subdirectory(graphData)
add_subdirectory(sequenceData)

function(addInput name)
  add_custom_target(data-${prefix}-${name} DEPENDS ${input}/${name})
  add_dependencies(more-suites-inputs-pbbs data-${prefix}-${name})
endfunction(addInput)
