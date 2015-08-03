add_library(lonestar STATIC src/BoilerPlate.cpp)
target_include_directories(lonestar PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(lonestar galois_runtime gllvm)
