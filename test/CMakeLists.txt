# Linking against the Boost libs via target_link_libraries
# can result in LLVM headers installed alongside Boost
# taking precedence over our vendored ones.
# avoid this by putting our vendored LLVM include paths
# first when compiling the executables that use them.
include_directories(BEFORE
  ${CMAKE_SOURCE_DIR}/libllvm/include
  ${CMAKE_CURRENT_BINARY_DIR}/../libllvm/include
)

makeTest(ADD_TARGET acquire DISTSAFE)
makeTest(ADD_TARGET bandwidth)
makeTest(ADD_TARGET barriers)
#makeTest(ADD_TARGET deterministic ${ROME})
makeTest(ADD_TARGET empty-member-lcgraph DISTSAFE)
makeTest(ADD_TARGET oneach)
#makeTest(ADD_TARGET filegraph DISTSAFE ${ROME})
makeTest(ADD_TARGET flatmap DISTSAFE EXP_OPT)
makeTest(ADD_TARGET forward-declare-graph DISTSAFE)
makeTest(ADD_TARGET foreach)
makeTest(ADD_TARGET gcollections DISTSAFE)
makeTest(ADD_TARGET graph-compile DISTSAFE)
makeTest(ADD_TARGET gslist)
makeTest(ADD_TARGET graph)
#makeTest(ADD_TARGET layergraph)
makeTest(ADD_TARGET lc-adaptor DISTSAFE)
makeTest(ADD_TARGET lock DISTSAFE)
makeTest(ADD_TARGET loop-overhead REQUIRES OPENMP_FOUND DISTSAFE)
makeTest(ADD_TARGET mem DISTSAFE)
makeTest(ADD_TARGET move DISTSAFE EXP_OPT)
makeTest(ADD_TARGET pc DISTSAFE)
#makeTest(ADD_TARGET sched DISTSAFE EXP_OPT)
makeTest(ADD_TARGET sort)
makeTest(ADD_TARGET static DISTSAFE)
makeTest(ADD_TARGET twoleveliteratora DISTSAFE)
makeTest(ADD_TARGET wakeup-overhead)
makeTest(ADD_TARGET worklists-compile DISTSAFE)
makeTest(ADD_TARGET floatingPointErrors)
makeTest(ADD_TARGET hwtopo DISTSAFE)
makeTest(ADD_TARGET morphgraph)
makeTest(ADD_TARGET papi)

#makeTest(TARGET lonestar/avi/AVIodgExplicitNoLock -n 0 -d 2 -f "${BASE}/inputs/avi/squareCoarse.NEU.gz")
#makeTest(TARGET lonestar/clustering/clustering -numPoints 1000)
#makeTest(TARGET lonestar/des/DESunordered "${BASE}/inputs/des/multTree6bit.net")
#makeTest(TARGET lonestar/kruskal/KruskalHand "${ROME}")
