function(mesh base numpoints)
  add_custom_command(OUTPUT ${base}.raw.node
    COMMAND python ${CMAKE_BINARY_DIR}/tools/generators/random-2d-points.py
      ${numpoints} 1 > ${base}.raw.node)
  add_custom_command(OUTPUT ${base}.node ${base}.ele ${base}.poly
    COMMAND delaunaytriangulation 
      -noverify -writemesh ${base} ${base}.raw.node 
    DEPENDS delaunaytriangulation ${base}.raw.node)
endfunction(mesh)

mesh(r500k  500000)
mesh(r1M   1000000)
mesh(r5M   5000000)
mesh(r10M 10000000)

add_custom_target(mesh-nodes DEPENDS r500k.node r1M.node r5M.node)
add_custom_target(mesh-eles DEPENDS r500k.node r500k.ele r500k.poly r1M.node r1M.ele r1M.poly r5M.node r5M.ele r5M.poly)

add_dependencies(more-inputs mesh-nodes mesh-eles)
