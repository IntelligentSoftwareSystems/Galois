# XXX(ddn): Broken for -O3
if(CMAKE_BUILD_TYPE MATCHES "Release")
  add_definitions(-O2)
endif()

executable(rMatGraph rMatGraph.C graphUtils.C)
executable(gridGraph gridGraph.C graphUtils.C)
executable(randLocalGraph randLocalGraph.C graphUtils.C)
executable(addWeights addWeights.C graphUtils.C)

function(galoisgraph base input)
  add_custom_command(OUTPUT ${input}/${base}.gr
    COMMAND graph-convert -pbbs2gr ${input}/${base} ${input}/${base}.gr
    DEPENDS graph-convert ${input}/${base})
  addInput(${base}.gr)
endfunction(galoisgraph)

function(rmat base edges numnodes numedges)
  if(${edges} MATCHES "YES")
    add_custom_command(OUTPUT ${input}/${base}
      COMMAND rMatGraph -a .5 -b .1 -m ${numedges} ${numnodes} ${input}/${base}
      DEPENDS rMatGraph)
  else()
    add_custom_command(OUTPUT ${input}/${base}
      COMMAND rMatGraph -j -a .5 -b .1 -m ${numedges} ${numnodes} ${input}/${base}
      DEPENDS rMatGraph)
    galoisgraph(${base} ${input})
  endif()
  addInput(${base})
endfunction(rmat)

function(rmatWeighted base edges numnodes numedges)
  rmat(${base}.raw ${edges} ${numnodes} ${numedges})
  add_custom_command(OUTPUT ${input}/${base}
    COMMAND addWeights ${input}/${base}.raw ${input}/${base}
    DEPENDS addWeights ${input}/${base}.raw)
  addInput(${base})
endfunction(rmatWeighted)

function(rand base edges numnodes numedges)
  if(${edges} MATCHES "YES")
    add_custom_command(OUTPUT ${input}/${base}
      COMMAND randLocalGraph -d 3 -m ${numedges} ${numnodes} ${input}/${base}
      DEPENDS randLocalGraph)
  else()
    add_custom_command(OUTPUT ${input}/${base}
      COMMAND randLocalGraph -j -d 3 -m ${numedges} ${numnodes} ${input}/${base}
      DEPENDS randLocalGraph)
    galoisgraph(${base} ${input})
  endif()
  addInput(${base})
endfunction(rand)

function(randWeighted base edges numnodes numedges)
  rand(${base}.raw ${edges} ${numnodes} ${numedges})
  add_custom_command(OUTPUT ${input}/${base}
    COMMAND addWeights ${input}/${base}.raw ${input}/${base}
    DEPENDS addWeights ${input}/${base}.raw)
  addInput(${base})
endfunction(randWeighted)

function(grid base edges dim numnodes)
  if(${edges} MATCHES "YES")
    add_custom_command(OUTPUT ${input}/${base}
      COMMAND gridGraph -d ${dim} ${numnodes} ${input}/${base}
      DEPENDS gridGraph)
  else()
    add_custom_command(OUTPUT ${input}/${base}
      COMMAND gridGraph -j -d ${dim} ${numnodes} ${input}/${base}
      DEPENDS gridGraph)
    galoisgraph(${base} ${input})
  endif()
  addInput(${base})
endfunction(grid)

function(gridWeighted base edges dim numnodes)
  grid(${base}.raw ${edges} ${dim} ${numnodes})
  add_custom_command(OUTPUT ${input}/${base}
    COMMAND addWeights ${input}/${base}.raw ${input}/${base}
    DEPENDS addWeights ${input}/${base}.raw)
  addInput(${base})
endfunction(gridWeighted)

