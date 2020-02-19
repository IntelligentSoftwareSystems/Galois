include_directories(BEFORE
  ${CMAKE_SOURCE_DIR}/libllvm/include
  ${CMAKE_CURRENT_BINARY_DIR}/../libllvm/include
)
include_directories(${CMAKE_SOURCE_DIR}/lonestargnn)
include_directories(${CMAKE_SOURCE_DIR}/libdeepgalois)

add_subdirectory(gcn)
