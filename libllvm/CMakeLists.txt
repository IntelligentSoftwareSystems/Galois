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

add_library(gllvm STATIC ${sources})
target_include_directories(gllvm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(gllvm PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/include)
include(llvm-extras)
