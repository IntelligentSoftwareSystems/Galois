###
# Common app generation
###

function(papp name)
  set(n "pbbs-${prefix}-${name}")
  addsuite(${n} ${ARGN})
endfunction(papp)

add_subdirectory(breadthFirstSearch)
#add_subdirectory(comparisonSort)
#add_subdirectory(convexHull)
add_subdirectory(delaunayRefine)
add_subdirectory(delaunayTriangulation)
#add_subdirectory(dictionary)
#add_subdirectory(integerSort)
add_subdirectory(maximalIndependentSet)
#add_subdirectory(maximalMatching)
add_subdirectory(minSpanningTree)
#add_subdirectory(nBody)
#add_subdirectory(nearestNeighbors)
#add_subdirectory(rayCast)
#add_subdirectory(removeDuplicates)
add_subdirectory(spanningTree)
#add_subdirectory(suffixArray)
