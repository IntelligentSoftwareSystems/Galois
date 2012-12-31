function(rgraph base numnodes)
  add_custom_command(OUTPUT ${base}.dimacs
    COMMAND python ${CMAKE_BINARY_DIR}/tools/generators/random-graph.py --density 4
      ${numnodes} 0 > ${base}.dimacs)
  add_custom_command(OUTPUT ${base}.gr
    COMMAND graph-convert -dimacs2gr ${base}.dimacs ${base}.gr
    DEPENDS ${base}.dimacs graph-convert)
endfunction(rgraph)

rgraph(r4-2e23 8388608) # powers of 2
rgraph(r4-2e24 16777216)
rgraph(r4-2e25 33554432) 
rgraph(r4-2e26 67108864) 

add_custom_target(more-random-graphs DEPENDS r4-2e23.gr r4-2e24.gr r4-2e25.gr r4-2e26.gr)
add_dependencies(more-inputs more-random-graphs)
