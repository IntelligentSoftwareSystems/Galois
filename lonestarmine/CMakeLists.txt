include_directories(BEFORE
  ${CMAKE_SOURCE_DIR}/libllvm/include
  ${CMAKE_CURRENT_BINARY_DIR}/../libllvm/include
)
include_directories(${CMAKE_SOURCE_DIR}/libpangolin/include)
add_subdirectory(tc)
add_subdirectory(kcl)
add_subdirectory(motif)
add_subdirectory(fsm)
add_subdirectory(sgl)

