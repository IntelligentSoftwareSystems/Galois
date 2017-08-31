set(LLVM_LINK_COMPONENTS Support)

add_clang_executable(clify
  clify.cpp
  )
target_link_libraries(clify
  clangTooling
  clangBasic
  clangASTMatchers
  )


#add_definitions(-DGEN_KERNEL=1)
add_clang_executable(clifyk
  clify.cpp
  )
target_compile_definitions(clifyk PRIVATE GEN_KERNEL=1)
target_link_libraries(clifyk
  clangTooling
  clangBasic
  clangASTMatchers
  )
  
if(UNIX)
  set(CLANGXX_LINK_OR_COPY create_symlink)
else()
  set(CLANGXX_LINK_OR_COPY copy)
endif()
