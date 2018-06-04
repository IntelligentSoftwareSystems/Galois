add_subdirectory(generators)
add_subdirectory(comparisons)
add_subdirectory(graph-convert)
add_subdirectory(dist-graph-convert)
add_subdirectory(graph-remap)
#add_subdirectory(graph-convert-standalone)
add_subdirectory(graph-stats)

include(ExternalProject)
find_program(WGET wget)
find_program(UNZIP unzip)

##########################################
# Broken external projects
##########################################

#ExternalProject_Add(cats-gens
#  PREFIX cats-gens
#  # LIST_SEPARATOR is replaced with ";" in generated Makefile
#  LIST_SEPARATOR |
#  # default is <prefix>/src
#  DOWNLOAD_DIR cats-gens/src/cats-gens
#  DOWNLOAD_COMMAND bash -c "if [ ! -e makefile ] | then ${WGET} --recursive --no-parent --no-host-directories --cut-dirs=3 --reject \"index.html*\" --quiet http://www.avglab.com/andrew/CATS/gens/ | fi"
#  PATCH_COMMAND bash -c "if [ ! -e CMakeLists.txt ] | then patch -p1 < ${CMAKE_CURRENT_SOURCE_DIR}/cats-gens.patch | fi"
#  CMAKE_ARGS 
#    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} 
#    -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS_RELEASE}
#    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
#    -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS_RELEASE}
#    -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}
#)

#ExternalProject_Add(snap
#  PREFIX snap
#  LIST_SEPARATOR |
#  DOWNLOAD_DIR snap/src/snap
#  DOWNLOAD_COMMAND bash -c "if [ ! -e Snap-11-12-31.zip ] | then ${WGET} --quiet http://snap.stanford.edu/snap/Snap-11-12-31.zip | fi"
#  UPDATE_COMMAND bash -c "if [ ! -e README ] | then ${UNZIP} Snap-11-12-31.zip | fi"
#  PATCH_COMMAND bash -c "if [ ! -e CMakeLists.txt ] | then patch -p1 < ${CMAKE_CURRENT_SOURCE_DIR}/snap.patch | fi"
#  CMAKE_ARGS 
#    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
#    -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS_RELEASE}
#    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
#    -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS_RELEASE}
#    -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}
#)

##########################################
# Working external projects
##########################################

# NB(ddn): bash -c "if [ ! -e ... " is a hack to prevent reexecuting
# commands when not needed

# Manually specify steps because ExternalProject doesn't know about unzip
ExternalProject_Add(triangle
  PREFIX triangle
  LIST_SEPARATOR |
  DOWNLOAD_DIR triangle/src/triangle
  DOWNLOAD_COMMAND bash -c "if [ ! -e triangle.zip ] | then ${WGET} --quiet http://www.netlib.org/voronoi/triangle.zip | fi"
  UPDATE_COMMAND bash -c "if [ ! -e makefile ] | then ${UNZIP} triangle.zip | fi"
  PATCH_COMMAND bash -c "if [ ! -e CMakeLists.txt ] | then patch -p1 < ${CMAKE_CURRENT_SOURCE_DIR}/triangle.patch | fi"
  CMAKE_ARGS 
    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} 
    -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS_RELEASE}
    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
    -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS_RELEASE}
    -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}
)

#set(more_tools hi_pr triangle cats-gens snap GTgraph-rmat SSCA2)
set(more_tools triangle GTgraph-rmat SSCA2)

# Simple download and patch
function(dl name url)
  ExternalProject_Add(${name}
    PREFIX ${name}
    LIST_SEPARATOR |
    DOWNLOAD_DIR ${name}/src/${name}
    URL ${url}
    PATCH_COMMAND bash -c "if [ ! -e CMakeLists.txt ] | then patch -p1 < ${CMAKE_CURRENT_SOURCE_DIR}/${name}.patch | fi"
    CMAKE_ARGS 
      -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} 
      -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS_RELEASE}
      -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
      -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS_RELEASE}
      -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}
  )
endfunction(dl)

function(cilkdl name url)
  if(HAVE_CILK)
    dl(${name} ${url})
    set(more_tools ${more_tools} ${name} PARENT_SCOPE)
  else()
    message(STATUS "NOT compiling ${name} (missing: CILK)")
  endif()
endfunction(cilkdl)

#dl(hi_pr http://www.avglab.com/andrew/soft/hipr.tar)
dl(GTgraph-rmat http://www.cse.psu.edu/~madduri/software/GTgraph/GTgraph.tar.gz)
dl(SSCA2 http://www.graphanalysis.org/benchmark/SSCA2-2.2.tar.gz)
cilkdl(bfs-schardl http://web.mit.edu/~neboat/www/code/bfs-icc-20120304.tar.gz)
cilkdl(bfs-pbbs http://www.cs.cmu.edu/~pbbs/benchmarks/breadthFirstSearch.tar)
cilkdl(delaunayrefinement-pbbs http://www.cs.cmu.edu/~pbbs/benchmarks/delaunayRefine.tar)
cilkdl(delaunaytriangulation-pbbs http://www.cs.cmu.edu/~pbbs/benchmarks/delaunayTriangulation.tar)
cilkdl(independentset-pbbs http://www.cs.cmu.edu/~pbbs/benchmarks/maximalIndependentSet.tar)
cilkdl(kruskal-pbbs http://www.cs.cmu.edu/~pbbs/benchmarks/minSpanningForest.tar)

add_custom_target(more-tools)
set_target_properties(${more_tools} PROPERTIES EXCLUDE_FROM_ALL true)
add_dependencies(more-tools ${more_tools})
