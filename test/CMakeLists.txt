function(makeTest name)
  add_executable(${name} ${name}.cpp)
  linkRuntime(${name})
  add_test(${name} ${name})
endfunction()

makeTest(compile-check)
makeTest(queue)
