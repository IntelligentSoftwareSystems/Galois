function(rgraph base numnodes numedges)
  add_custom_command(OUTPUT ${base}.rmat
    COMMAND ${CMAKE_BINARY_DIR}/tools/bin/GTgraph-rmat -n ${numnodes} -m ${numedges}
      -o ${base}.rmat
    DEPENDS ${CMAKE_BINARY_DIR}/tools/bin/GTgraph-rmat)
  add_custom_command(OUTPUT ${base}.gr
    COMMAND ${CMAKE_BINARY_DIR}/tools/graph-convert/graph-convert
      ${base}.rmat ${base}.gr
    DEPENDS ${base}.rmat graph-convert)
endfunction(rgraph)

rgraph(rmat8-2e12 4096 32768) 
rgraph(rmat8-2e13 8192 65536) 
rgraph(rmat8-2e14 16384 131072) 

add_custom_target(more-scalefree-graphs DEPENDS rmat8-2e12.gr rmat8-2e13.gr rmat8-2e14.gr)
add_dependencies(more-inputs more-scalefree-graphs)
