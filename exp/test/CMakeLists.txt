function(makeTest name)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} galois)
  add_test(${name} ${name})
endfunction()

add_definitions(-O0)

makeTest(pairingheap)
makeTest(skiplist)
