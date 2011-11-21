function(mesh base numpoints)
  add_custom_command(OUTPUT ${base}.node
    COMMAND ${CMAKE_BINARY_DIR}/tools/generators/random-2d-points.py
      ${numpoints} 1 > ${base}.node)
  add_custom_command(OUTPUT ${base}.1.node ${base}.1.ele ${base}.1.poly
    COMMAND ${CMAKE_BINARY_DIR}/apps/delaunaytriangulation/delaunaytriangulation
      -noverify -writemesh ${base}.1 ${base}.node 
    DEPENDS delaunaytriangulation ${base}.node)
endfunction(mesh)

mesh(r500k 500000)
mesh(r1M  1000000)
mesh(r5M  5000000)

add_custom_target(mesh-nodes DEPENDS r500k.node r1M.node r5M.node)
add_custom_target(mesh-eles DEPENDS r500k.1.node r500k.1.ele r500k.1.poly r1M.1.node r1M.1.ele r1M.1.poly r5M.1.node r5M.1.ele r5M.1.poly)

add_dependencies(more-inputs mesh-nodes mesh-eles)
