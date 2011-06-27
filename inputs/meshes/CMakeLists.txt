find_program(PYTHON python)
add_custom_command(OUTPUT r500k.node
  COMMAND ${PYTHON} ../../tools/random-2d-points/random-2d-points.py 500000 1 > r500k.node)
add_custom_command(OUTPUT r1M.node
  COMMAND ${PYTHON} ../../tools/random-2d-points/random-2d-points.py 1000000 1 > r1M.node)
add_custom_command(OUTPUT r5M.node
  COMMAND ${PYTHON} ../../tools/random-2d-points/random-2d-points.py 5000000 1 > r5M.node)
add_custom_command(OUTPUT r500k.1.node r500k.1.ele r500k.1.poly
  COMMAND ../../apps/delaunaytriangulation/delaunaytriangulation -noverify -writemesh r500k.node)
add_custom_command(OUTPUT r1M.1.node r1M.1.ele r1M.1.poly
  COMMAND ../../apps/delaunaytriangulation/delaunaytriangulation -noverify -writemesh r1M.node)
add_custom_command(OUTPUT r5M.1.node r5M.1.ele r5M.1.poly
  COMMAND ../../apps/delaunaytriangulation/delaunaytriangulation -noverify -writemesh r5M.node)

add_custom_target(mesh-nodes DEPENDS r500k.node r1M.node r5M.node)
add_custom_target(mesh-eles DEPENDS r500k.1.node r500k.1.ele r500k.1.poly r1M.1.node r1M.1.ele r1M.1.poly r5M.1.node r5M.1.ele r5M.1.poly)
add_dependencies(mesh-eles mesh-nodes delaunaytriangulation)

add_dependencies(more-inputs mesh-nodes mesh-eles)
