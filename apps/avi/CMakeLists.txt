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

find_package(Fortran)
if(Fortran_FOUND)
  enable_language(Fortran OPTIONAL)
endif()

find_package(LAPACK)
find_package(ZLIB)

if(ZLIB_FOUND AND NOT ARCH STREQUAL "mic")
  add_definitions(-DGALOIS_HAS_ZLIB)
endif()

# TODO(ddn): CMake Boost module on BG/Q cannot be called multiple times. Skip
# avi althogether for now.
if(CMAKE_SYSTEM_NAME MATCHES "BlueGeneQ")
  set(Boost_FOUND off)
else()
  # Needs a slightly higher version than other Galois apps
  find_package(Boost 1.50.0 COMPONENTS system iostreams filesystem)
endif()

if(CMAKE_Fortran_COMPILER_WORKS AND LAPACK_FOUND AND Boost_FOUND)
  include_directories(${ZLIB_INCLUDE_DIRS})
  add_library(AVI ${Sources})

  if (ZLIB_FOUND AND NOT ARCH STREQUAL "mic") 
    target_link_libraries(AVI ${ZLIB_LIBRARIES})
    target_link_libraries(AVI ${Boost_LIBRARIES})
  endif()

  set(AVI_FOUND TRUE)
endif()

app(AVIorderedSerial main/AVIorderedSerial.cpp
  REQUIRES AVI_FOUND
  EXTLIBS AVI ${LAPACK_LIBRARIES})

#app(AVIodgOrdered main/AVIodgOrdered.cpp 
#  REQUIRES AVI_FOUND 
#  EXTLIBS AVI ${LAPACK_LIBRARIES})

app(AVIodgExplicit main/AVIodgExplicit.cpp
  REQUIRES AVI_FOUND
  EXTLIBS AVI ${LAPACK_LIBRARIES})

app(AVIodgExplicitNoLock main/AVIodgExplicitNoLock.cpp
  REQUIRES AVI_FOUND
  EXTLIBS AVI ${LAPACK_LIBRARIES})

# the files may be removed from the release without causing error in cmake
if(0)
  app(AVIodgReadonly exp/AVIodgReadonly.cpp 
    REQUIRES AVI_FOUND USE_EXP
    EXTLIBS AVI ${LAPACK_LIBRARIES})

  app(AVIlevelExec exp/AVIlevelExec.cpp 
    REQUIRES LAPACK_FOUND AVI_FOUND USE_EXP
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
