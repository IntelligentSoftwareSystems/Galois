file(GLOB Sources 
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
  dummy.f90
)

enable_language(Fortran OPTIONAL)
find_package(LAPACK)
find_package(ZLIB)
# TODO(ddn): CMake Boost module on BG/Q cannot be called multiple times. Skip
# avi althogether for now.
if(CMAKE_SYSTEM_NAME MATCHES "BlueGeneQ")
  set(Boost_FOUND off)
else()
  find_package(Boost 1.36.0 COMPONENTS system iostreams filesystem)
endif()

if(CMAKE_Fortran_COMPILER_WORKS AND LAPACK_FOUND AND ZLIB_FOUND AND Boost_FOUND)
  include_directories(${ZLIB_INCLUDE_DIRS})
  add_library(AVI ${Sources})
  target_link_libraries(AVI ${ZLIB_LIBRARIES})
  target_link_libraries(AVI ${Boost_LIBRARIES})
  set(AVI_FOUND TRUE)
endif()

app(AVIorderedSerial main/AVIorderedSerial.cpp
  REQUIRES AVI_FOUND
  EXTLIBS AVI ${LAPACK_LIBRARIES})

app(AVIodgOrdered main/AVIodgOrdered.cpp 
  REQUIRES AVI_FOUND 
  EXTLIBS AVI ${LAPACK_LIBRARIES})

app(AVIodgExplicit main/AVIodgExplicit.cpp 
  REQUIRES AVI_FOUND
  EXTLIBS AVI ${LAPACK_LIBRARIES})

app(AVIodgExplicitNoLock main/AVIodgExplicitNoLock.cpp
  REQUIRES AVI_FOUND
  EXTLIBS AVI ${LAPACK_LIBRARIES})

# the files may be removed from the release without causing error in cmake
#if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/exp)
if(0)
  app(AVIodgReadonly exp/AVIodgReadonly.cpp 
    REQUIRES AVI_FOUND USE_EXP
    EXTLIBS AVI ${LAPACK_LIBRARIES})

  app(AVIodgImplicit exp/AVIodgImplicit.cpp 
    REQUIRES AVI_FOUND USE_EXP
    EXTLIBS AVI ${LAPACK_LIBRARIES})

  app(AVIodgAutoPriLock exp/AVIodgAutoPriLock.cpp 
    REQUIRES AVI_FOUND USE_EXP
    EXTLIBS AVI ${LAPACK_LIBRARIES})

  app(AVIodgAutoShare exp/AVIodgAutoShare.cpp 
    REQUIRES AVI_FOUND USE_EXP
    EXTLIBS AVI ${LAPACK_LIBRARIES})

  app(AVIodgNB exp/AVIodgNB.cpp 
    REQUIRES AVI_FOUND USE_EXP
    EXTLIBS AVI ${LAPACK_LIBRARIES})
endif()

include_directories(util)
include_directories(main)
include_directories(libElm)
include_directories(libElm/libQuad)
include_directories(libElm/libGeom)
include_directories(libElm/libShapesEvaluated)
include_directories(libElm/libShape)
include_directories(libElm/libElement)
include_directories(libAVI)
include_directories(libMeshInit)
include_directories(libMeshInit/dgmechanics)
include_directories(libMeshInit/femap)
include_directories(libMat)
include_directories(libElOp)
