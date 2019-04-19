set(sources
		src/graph.cc
		src/orbit.cc
		src/partition.cc
)

add_library(bliss STATIC ${sources})
target_include_directories(bliss PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(bliss ${Boost_SERIALIZATION_LIBRARY_RELEASE})

if (CMAKE_HAVE_PTHREAD_H)
   add_definitions(-DGALOIS_HAVE_PTHREAD)
   target_link_libraries(bliss Threads::Threads)
endif()

add_library(bliss_obj OBJECT ${sources})
target_include_directories(bliss_obj PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
set_target_properties (bliss_obj PROPERTIES 
  INTERFACE_POSITION_INDEPENDENT_CODE On 
  POSITION_INDEPENDENT_CODE On
)
