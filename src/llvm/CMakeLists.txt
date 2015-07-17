project(gLLVM)

add_library(gllvm OBJECT
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

target_include_directories(gllvm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

include(llvm-extras)
