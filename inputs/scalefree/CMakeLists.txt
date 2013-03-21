function(rgraph base numnodes numedges)
  add_custom_command(OUTPUT ${base}.dimacs
    COMMAND python ${CMAKE_BINARY_DIR}/tools/generators/rmat.py
      ${CMAKE_BINARY_DIR}/tools/bin/GTgraph-rmat -n ${numnodes} -m ${numedges} ${base}.rmat
    DEPENDS ${CMAKE_BINARY_DIR}/tools/bin/GTgraph-rmat)
  add_custom_command(OUTPUT ${base}.gr
    COMMAND graph-convert-standalone -dimacs2gr ${base}.dimacs ${base}.gr
    DEPENDS ${base}.dimacs graph-convert-standalone)
endfunction(rgraph)

rgraph(rmat8-2e14 16384 131072) 
rgraph(rmat8-2e18 262144 2097152)
rgraph(rmat8-2e26 67108864 536870912) 

add_custom_target(more-scalefree-graphs DEPENDS rmat8-2e14.gr rmat8-2e18.gr rmat8-2e26.gr)
add_dependencies(more-inputs more-scalefree-graphs)
add_dependencies(more-scalefree-graphs more-tools)
