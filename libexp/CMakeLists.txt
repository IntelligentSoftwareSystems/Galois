file(GLOB sources src/*.cpp)
# set(sources src/DoAllWrap.cpp)

add_library(galois_exp STATIC ${sources})
target_include_directories(galois_exp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)


