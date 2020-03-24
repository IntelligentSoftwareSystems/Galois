# Linking against the Boost libs via target_link_libraries
# can result in LLVM headers installed alongside Boost
# taking precedence over our vendored ones.
# avoid this by putting our vendored LLVM include paths
# first when compiling the executables that use them.
include_directories(BEFORE
  ${CMAKE_SOURCE_DIR}/libllvm/include
  ${CMAKE_CURRENT_BINARY_DIR}/../libllvm/include
)

add_test_unit(ADD_TARGET acquire DISTSAFE)
add_test_unit(ADD_TARGET bandwidth)
add_test_unit(ADD_TARGET barriers)
add_test_unit(ADD_TARGET empty-member-lcgraph DISTSAFE)
add_test_unit(ADD_TARGET flatmap DISTSAFE EXP_OPT)
add_test_unit(ADD_TARGET floatingPointErrors)
add_test_unit(ADD_TARGET foreach)
add_test_unit(ADD_TARGET forward-declare-graph DISTSAFE)
add_test_unit(ADD_TARGET gcollections DISTSAFE)
add_test_unit(ADD_TARGET graph)
add_test_unit(ADD_TARGET graph-compile DISTSAFE)
add_test_unit(ADD_TARGET gslist)
add_test_unit(ADD_TARGET gtuple)
add_test_unit(ADD_TARGET hwtopo DISTSAFE)
add_test_unit(ADD_TARGET lc-adaptor DISTSAFE)
add_test_unit(ADD_TARGET lock DISTSAFE)
add_test_unit(ADD_TARGET lockmgr)
add_test_unit(ADD_TARGET loop-overhead REQUIRES OPENMP_FOUND DISTSAFE)
add_test_unit(ADD_TARGET mem DISTSAFE)
add_test_unit(ADD_TARGET morphgraph)
add_test_unit(ADD_TARGET move DISTSAFE EXP_OPT)
add_test_unit(ADD_TARGET oneach)
add_test_unit(ADD_TARGET papi 2)
add_test_unit(ADD_TARGET pc DISTSAFE)
add_test_unit(ADD_TARGET sort)
add_test_unit(ADD_TARGET static DISTSAFE)
add_test_unit(ADD_TARGET twoleveliteratora DISTSAFE)
add_test_unit(ADD_TARGET wakeup-overhead)
add_test_unit(ADD_TARGET worklists-compile DISTSAFE)
