function(makeTest name)
  add_executable(${name} ${name}.cpp)
  linkRuntime(${name})
  add_test(${name} ${name})
endfunction()

add_definitions(-O0)

makeTest(printhost)
#makeTest(pairingheap)
#makeTest(skiplist)
#makeTest(dist)
#makeTest(distnodes)
#makeTest(foreachtask)
makeTest(networktest)

