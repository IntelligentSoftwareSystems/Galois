set(sources
                 src/Allocator.cpp
                 src/APFloat.cpp
                 src/APInt.cpp
                 src/CommandLine.cpp
                 src/SmallPtrSet.cpp
                 src/SmallVector.cpp
                 src/StringMap.cpp
                 src/StringRef.cpp
                 src/Twine.cpp
)

add_library(gllvm_obj OBJECT ${sources})
target_include_directories(gllvm_obj BEFORE PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(gllvm_obj BEFORE PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/include)

add_library(gllvm STATIC $<TARGET_OBJECTS:gllvm_obj>)
target_include_directories(gllvm BEFORE PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(gllvm BEFORE PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/include)
include(llvm-extras)

