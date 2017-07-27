# If we don't need RTTI or EH, there's no reason to export anything
# from the plugin.
if( NOT MSVC ) # MSVC mangles symbols differently, and
                # PrintFunctionNames.export contains C++ symbols.
  if( NOT LLVM_REQUIRES_RTTI )
    if( NOT LLVM_REQUIRES_EH )
      set(LLVM_EXPORTED_SYMBOL_FILE
         ${CMAKE_CURRENT_SOURCE_DIR}/AosToSoaPlugin.exports)
    endif()
  endif()
endif()

add_llvm_loadable_module(AosToSoaPlugin AosToSoaPlugin.cpp)

target_link_libraries(AosToSoaPlugin ${cmake_2_8_12_PRIVATE} clangASTMatchers)

if(LLVM_ENABLE_PLUGINS AND (WIN32 OR CYGWIN))
   target_link_libraries(AosToSoaPlugin ${cmake_2_8_12_PRIVATE}
    clangASTMatchers
    clangAST
    clangBasic
    clangFrontend
    LLVMSupport
    )
endif()
