add_custom_target(more-suites)
function(addsuite target)
  add_dependencies(more-suites ${target})
  set_target_properties(${target} PROPERTIES EXCLUDE_FROM_ALL true)
endfunction()

add_subdirectory(pbbs-0.1)

