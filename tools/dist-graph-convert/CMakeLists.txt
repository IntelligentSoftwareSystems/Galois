if(ENABLE_DIST_GALOIS) 
  app(dist-graph-convert dist-graph-convert.cpp DISTSAFE)
  target_include_directories(dist-graph-convert PUBLIC ${CMAKE_SOURCE_DIR}/libdist/include)
  target_include_directories(dist-graph-convert PUBLIC ${LWCI_INCLUDE})

  add_library(dist-graph-convert-help STATIC dist-graph-convert-helpers.cpp)
  target_include_directories(dist-graph-convert-help PUBLIC 
    ${CMAKE_SOURCE_DIR}/libllvm/include
    ${CMAKE_SOURCE_DIR}/libgalois/include
    ${CMAKE_SOURCE_DIR}/libdist/include
    ${CMAKE_SOURCE_DIR}/tools/dist-graph-convert
    ${LWCI_INCLUDE}
  )

  target_link_libraries(dist-graph-convert galois_dist dist-graph-convert-help)
  target_include_directories(dist-graph-convert PUBLIC ${CMAKE_SOURCE_DIR}/libdist/include)
endif()
