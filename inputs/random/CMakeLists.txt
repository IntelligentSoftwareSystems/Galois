function(rgraph base numnodes)
  add_custom_command(OUTPUT ${base}.dimacs
    COMMAND ${CMAKE_BINARY_DIR}/tools/generators/random-graph.py
      ${numnodes} 0 > ${base}.dimacs)
  add_custom_command(OUTPUT ${base}.gr
    COMMAND ${CMAKE_BINARY_DIR}/tools/graph-convert/graph-convert
      ${base}.dimacs ${base}.gr
    DEPENDS ${base}.dimacs graph-convert)
endfunction(rgraph)

rgraph(r4-2e23 8388608)
rgraph(r4-2e24 16777216)
rgraph(r4-2e25 33554432) # powers of 2

add_custom_target(more-random-graphs DEPENDS r4-2e23.gr r4-2e24.gr r4-2e25.gr)
add_dependencies(more-inputs more-random-graphs)
