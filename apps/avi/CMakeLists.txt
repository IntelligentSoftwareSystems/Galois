
file (GLOB Sources 
util/*.cpp
libMat/*.cpp
libElm/libGeom/*.cpp
libElm/libQuad/*.cpp
libElm/libShape/*.cpp
libElm/libShapesEvaluated/*.cpp
libElm/libElement/*.cpp
libElm/*.cpp
libElOp/*.cpp
libAVI/*.cpp
libMeshInit/femap/*.cpp
libMeshInit/dgmechanics/*.cpp
libMeshInit/*.cpp
)

add_executable(AVIorderedSerial main/AVIorderedSerial.cpp  ${Sources})
target_link_libraries(AVIorderedSerial galois) 
target_link_libraries(AVIorderedSerial ${CMAKE_THREAD_LIBS_INIT}) 
target_link_libraries(AVIorderedSerial hoard)
target_link_libraries(AVIorderedSerial lapack)

add_executable(AVIunordered main/AVIunordered.cpp  ${Sources})
target_link_libraries(AVIunordered galois) 
target_link_libraries(AVIunordered ${CMAKE_THREAD_LIBS_INIT}) 
target_link_libraries(AVIunordered hoard)
target_link_libraries(AVIunordered lapack)



include_directories (libElm)
include_directories (libElm/libQuad)
include_directories (libElm/libGeom)
include_directories (libElm/libShapesEvaluated)
include_directories (libElm/libShape)
include_directories (libElm/libElement)
include_directories (util)
include_directories (main)
include_directories (libAVI)
include_directories (libMeshInit)
include_directories (libMeshInit/dgmechanics)
include_directories (libMeshInit/femap)
include_directories (libMat)
include_directories (libElOp)
