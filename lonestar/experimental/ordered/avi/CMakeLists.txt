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

find_package(Fortran)
if(Fortran_FOUND)
  enable_language(Fortran OPTIONAL)
endif()

find_package(LAPACK QUIET)
find_package(ZLIB)
find_package(Boost 1.50.0 COMPONENTS system iostreams filesystem)

# XXX: Disabling reading of GZ input files for now. 
# if(Boos_FOUND AND ZLIB_FOUND AND NOT ARCH STREQUAL "mic")
  # add_definitions(-DGALOIS_HAS_GZ_SUPP)
  # set(HAS_GZ_SUPP TRUE)
# endif()
set(HAS_GZ_SUPP FALSE)


if(CMAKE_Fortran_COMPILER_WORKS AND LAPACK_FOUND)
  include_directories(${ZLIB_INCLUDE_DIRS})
  add_library(AVI ${Sources})

  if (HAS_GZ_SUPP) 
    target_link_libraries(AVI ${ZLIB_LIBRARIES})
    target_link_libraries(AVI ${Boost_LIBRARIES})
  endif()

  target_link_libraries(AVI galois_runtime)
  target_link_libraries(AVI galois_substrate)

  set(AVI_FOUND TRUE)
endif()

if (AVI_FOUND)
  app(AVIorderedSerial main/AVIorderedSerial.cpp
    REQUIRES AVI_FOUND
    EXTLIBS AVI ${LAPACK_LIBRARIES})

  app(AVIodgExplicit main/AVIodgExplicit.cpp
    REQUIRES AVI_FOUND
    EXTLIBS AVI ${LAPACK_LIBRARIES})

  app(AVIodgExplicitNoLock main/AVIodgExplicitNoLock.cpp
    REQUIRES AVI_FOUND
    EXTLIBS AVI ${LAPACK_LIBRARIES})

  # the files may be removed from the release without causing error in cmake
  if(USE_EXP)
    app(AVIodgOrdered main/AVIodgOrdered.cpp 
      REQUIRES AVI_FOUND
      EXTLIBS AVI ${LAPACK_LIBRARIES})

    app(AVIorderedSpec exp/AVIorderedSpec.cpp 
      REQUIRES AVI_FOUND
      EXTLIBS AVI ${LAPACK_LIBRARIES})

    app(AVIlevelExec exp/AVIlevelExec.cpp 
      REQUIRES AVI_FOUND
      EXTLIBS AVI ${LAPACK_LIBRARIES})

    app(AVIserialGalloc exp/AVIserialGalloc.cpp 
      REQUIRES AVI_FOUND
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

  add_test_scale(web   AVIodgExplicitNoLock -d 2 -n 1 -e 0.1 -f "${BASEINPUT}/avi/10x10_42k.NEU.gz")
  add_test_scale(small AVIodgExplicitNoLock -d 2 -n 0 -f "${BASEINPUT}/avi/squareCoarse.NEU.gz")

endif()
