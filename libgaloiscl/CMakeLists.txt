set(sources
        src/CL_Header.cpp
)
include_directories(${OPENCL_INCLUDE_DIRS})
add_library(galois_cl STATIC ${sources})
target_include_directories(galois_cl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(galois_cl OpenCL)