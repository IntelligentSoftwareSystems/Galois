add_library(distbench STATIC src/Start.cpp src/Input.cpp src/Output.cpp)
target_include_directories(distbench PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

find_package(Arrow CONFIG REQUIRED)
if (${ARROW_VERSION} VERSION_LESS 0.17 OR ${ARROW_VERSION} VERSION_GREATER_EQUAL 0.18)
  message(FATAL_ERROR "libarrow must be version 0.17.*; found ${ARROW_VERSION} instead")
endif()
get_filename_component(ARROW_CONFIG_DIR ${Arrow_CONFIG} DIRECTORY)

find_package(Parquet HINTS ${ARROW_CONFIG_DIR})
target_link_libraries(distbench Galois::cusp Galois::gluon LLVMSupport)
target_link_libraries(distbench tsuba)
target_link_libraries(distbench arrow_shared parquet_shared)
