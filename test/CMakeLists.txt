# Linking against the Boost libs via target_link_libraries
# can result in LLVM headers installed alongside Boost
# taking precedence over our vendored ones.
# avoid this by putting our vendored LLVM include paths
# first when compiling the executables that use them.
include_directories(BEFORE
  ${CMAKE_SOURCE_DIR}/libllvm/include
  ${CMAKE_CURRENT_BINARY_DIR}/../libllvm/include
)

add_test_unit(ADD_TARGET acquire)
add_test_unit(ADD_TARGET bandwidth)
add_test_unit(ADD_TARGET barriers)
add_test_unit(ADD_TARGET empty-member-lcgraph)
add_test_unit(ADD_TARGET flatmap)
add_test_unit(ADD_TARGET floatingPointErrors)
add_test_unit(ADD_TARGET foreach)
add_test_unit(ADD_TARGET forward-declare-graph)
add_test_unit(ADD_TARGET gcollections)
add_test_unit(ADD_TARGET graph)
add_test_unit(ADD_TARGET graph-compile)
add_test_unit(ADD_TARGET gslist)
add_test_unit(ADD_TARGET gtuple)
add_test_unit(ADD_TARGET hwtopo)
add_test_unit(ADD_TARGET lc-adaptor)
add_test_unit(ADD_TARGET lock)
add_test_unit(ADD_TARGET loop-overhead REQUIRES OPENMP_FOUND)
add_test_unit(ADD_TARGET mem)
add_test_unit(ADD_TARGET morphgraph)
add_test_unit(ADD_TARGET move)
add_test_unit(ADD_TARGET oneach)
add_test_unit(ADD_TARGET papi 2)
add_test_unit(ADD_TARGET pc )
add_test_unit(ADD_TARGET sort)
add_test_unit(ADD_TARGET static)
add_test_unit(ADD_TARGET twoleveliteratora)
add_test_unit(ADD_TARGET wakeup-overhead)
add_test_unit(ADD_TARGET worklists-compile)
