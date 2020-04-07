app(dist-graph-convert dist-graph-convert.cpp)
add_library(dist-graph-convert-help STATIC dist-graph-convert-helpers.cpp)
target_link_libraries(dist-graph-convert-help galois_dist_async)

target_include_directories(dist-graph-convert-help PUBLIC
  ${CMAKE_SOURCE_DIR}/libllvm/include
  ${CMAKE_SOURCE_DIR}/libgalois/include
  ${CMAKE_SOURCE_DIR}/libdist/include
  ${CMAKE_SOURCE_DIR}/tools/dist-graph-convert
  ${LWCI_INCLUDE}
)

target_link_libraries(dist-graph-convert dist-graph-convert-help)
