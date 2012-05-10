executable(uniform uniform.C)
executable(plummer plummer.C)
executable(addRays addRays.C)

function(inSphere base dim num)
  add_custom_command(OUTPUT ${input}/${base}
    COMMAND uniform -s -d ${dim} ${num} ${input}/${base}
    DEPENDS uniform)
  addInput(${base})
endfunction(inSphere)

function(onSphere base dim num)
  add_custom_command(OUTPUT ${input}/${base}
    COMMAND uniform -S -d ${dim} ${num} ${input}/${base}
    DEPENDS uniform)
  addInput(${base})
endfunction(onSphere)

function(inCube base dim num)
  add_custom_command(OUTPUT ${input}/${base}
    COMMAND uniform -d ${dim} ${num} ${input}/${base}
    DEPENDS uniform)
  addInput(${base})
endfunction(inCube)

function(kuzmin base dim num)
  add_custom_command(OUTPUT ${input}/${base}
    COMMAND plummer -d ${dim} ${num} ${input}/${base}
    DEPENDS uniform)
  addInput(${base})
endfunction(kuzmin)

function(mesh base num)
  add_custom_command(OUTPUT ${input}/${base}.points
    COMMAND uniform -d 2 ${num} ${input}/${base}.points
    DEPENDS uniform)
  add_custom_command(OUTPUT ${input}/${base}
    COMMAND pbbs-delaunayTriangulation-incrementalDelaunay -o ${input}/${base} ${input}/${base}.points
    DEPENDS pbbs-delaunayTriangulation-incrementalDelaunay ${input}/${base}.points)
  addInput(${base})
endfunction(mesh)

find_program(BZCAT bzcat)

function(genRays triangles rays)
  add_custom_command(OUTPUT ${input}/${triangles}
    COMMAND bzcat ${input_src}/geometryData/data/${triangles}.bz2 > ${input}/${triangles})
  add_custom_command(OUTPUT ${input}/${rays}
    COMMAND addRays ${input}/${triangles} ${input}/${rays}
    DEPENDS addRays ${input}/${triangles})
  addInput(${rays})
  addInput(${triangles})
endfunction(genRays)

