function(makeTest name)
  add_executable(${name} ${name}.cpp)
  linkRuntime(${name})
  add_test(${name} ${name})
endfunction()

add_definitions(-O0)

makeTest(addhost)
makeTest(printhost)
#makeTest(pairingheap)
#makeTest(skiplist)
#makeTest(dist)
makeTest(distnodes)
makeTest(distgraph)
#makeTest(foreachtask)
makeTest(networktest)

