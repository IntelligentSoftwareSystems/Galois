add_custom_target(more-suites)
function(addsuite target)
  target_link_libraries(${target} ${GALOIS_EXP_LIBRARIES})
  add_dependencies(more-suites ${target})
  set_target_properties(${target} PROPERTIES EXCLUDE_FROM_ALL true)
  linkRuntime(${target})
endfunction()

add_custom_target(more-suites-inputs)

add_subdirectory(pbbs-0.1)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/parsec-2.1")
  add_subdirectory(parsec-2.1)
else()
  message("Missing parsec-2.1 suite; run download_external.sh to download")
endif()
