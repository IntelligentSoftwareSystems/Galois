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
  dummy.f95
)

add_library(AVI ${Sources})

app(AVIorderedSerial main/AVIorderedSerial.cpp
  REQUIRES LAPACK_FOUND ZLIB_FOUND
  EXTLIBS AVI ${LAPACK_LIBRARIES})


app(AVIodgOrdered main/AVIodgOrdered.cpp 
  REQUIRES LAPACK_FOUND ZLIB_FOUND
  EXTLIBS AVI ${LAPACK_LIBRARIES})

# the files may be removed from the release without causing error in cmake
#if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/exp)
if(0)
  app(AVIodgExplicit exp/AVIodgExplicit.cpp 
    REQUIRES LAPACK_FOUND ZLIB_FOUND
    EXTLIBS AVI ${LAPACK_LIBRARIES})

  app(AVIodgExplicitNoLock exp/AVIodgExplicitNoLock.cpp
    REQUIRES LAPACK_FOUND ZLIB_FOUND
    EXTLIBS AVI ${LAPACK_LIBRARIES})

  app(AVIodgReadonly exp/AVIodgReadonly.cpp 
    REQUIRES LAPACK_FOUND ZLIB_FOUND USE_EXP
    EXTLIBS AVI ${LAPACK_LIBRARIES})

  app(AVIodgImplicit exp/AVIodgImplicit.cpp 
    REQUIRES LAPACK_FOUND ZLIB_FOUND USE_EXP
    EXTLIBS AVI ${LAPACK_LIBRARIES})

  app(AVIodgAutoPriLock exp/AVIodgAutoPriLock.cpp 
    REQUIRES LAPACK_FOUND ZLIB_FOUND USE_EXP
    EXTLIBS AVI ${LAPACK_LIBRARIES})

  app(AVIodgAutoShare exp/AVIodgAutoShare.cpp 
    REQUIRES LAPACK_FOUND ZLIB_FOUND USE_EXP
    EXTLIBS AVI ${LAPACK_LIBRARIES})

  app(AVIodgNB exp/AVIodgNB.cpp 
    REQUIRES LAPACK_FOUND ZLIB_FOUND USE_EXP
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
