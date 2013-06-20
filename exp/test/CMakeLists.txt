function(makeTest name)
  add_executable(test-${name} ${name}.cpp)
  target_link_libraries(test-${name} galois)
  add_test(${name} test-${name})
endfunction()

makeTest(pairingheap)
makeTest(skiplist)
