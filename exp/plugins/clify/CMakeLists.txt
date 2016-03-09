set(LLVM_LINK_COMPONENTS Support)

add_clang_executable(clify
  clify.cpp
  )
target_link_libraries(clify
  clangTooling
  clangBasic
  clangASTMatchers
  )

if(UNIX)
  set(CLANGXX_LINK_OR_COPY create_symlink)
else()
  set(CLANGXX_LINK_OR_COPY copy)
endif()
