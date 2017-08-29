set(LLVM_LINK_COMPONENTS Support)

add_clang_executable(distifier
  distifier.cpp
  )
target_link_libraries(distifier
  clangTooling
  clangBasic
  clangASTMatchers
  )

if(UNIX)
  set(CLANGXX_LINK_OR_COPY create_symlink)
else()
  set(CLANGXX_LINK_OR_COPY copy)
endif()
