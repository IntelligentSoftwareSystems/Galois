cmake_minimum_required(VERSION 2.6)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")
project(Galois)
set(GALOIS_VERSION_MAJOR "2")
set(GALOIS_VERSION_MINOR "1")
set(GALOIS_VERSION_PATCH "5")
set(GALOIS_COPYRIGHT_YEAR "2012") # Also in COPYRIGHT

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "No build type selected, default to Release")
  set(CMAKE_BUILD_TYPE "Release")
endif()

###### Options (alternatively pass as options to cmake -DName=Value) ######

#use profiling
set(USE_GPROF OFF CACHE BOOL "Enable GCC profiling")

#Enable iss specific options; should be OFF in the general release; all guarded by USE_EXP
set(USE_EXP OFF CACHE BOOL "Use experimental features") 
set(USE_VTUNE ON CACHE BOOL "Use vtune (if found)")
set(EXP_DOALL "PTHREAD" CACHE STRING "Which type of implementation of parallel_doall")
set(USE_PROF OFF CACHE BOOL "Use profiling specific features")

###### Configure (users don't need to go beyond here) ######
enable_testing()

#ICC
if(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -wd68 -wd981 -wd383 -wd869 -wd2196 -wd279")
endif()

message(STATUS "Using ${CMAKE_CXX_COMPILER_ID}")
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -isystem=/net/faraday/workspace/local/modules/gcc-4.7")
endif()

#PThreads
find_package(Threads)
set(RT_LINK ${CMAKE_THREAD_LIBS_INIT} ${RT_LINK})

#NUMA (linux)
find_package(LIBNUMA)
if(LIBNUMA_FOUND)
  set(RT_LINK numa ${RT_LINK})
  add_definitions(-DGALOIS_USE_NUMA)
endif()

#LAPACK
enable_language(Fortran)
find_package(LAPACK)

#CILK
include(CheckCilk)

# ZLIB
find_package(ZLIB REQUIRED)
if(ZLIB_FOUND)
  include_directories(${ZLIB_INCLUDE_DIRS})
  set(RT_LINK ${ZLIB_LIBRARIES} ${RT_LINK})
endif(ZLIB_FOUND)

#Boost
#not needed now but will be needed for newer version of Boost after 1.4X ish
set(Boost_ADDITIONAL_VERSIONS "1.40" "1.40.0" "1.47" "1.47.0" "1.49" "1.49.0" "1.51.0")
set(Boost_USE_MULTITHREADED OFF) # newer boost builds don't use -mt extension
#Other options that can be used with find_package(Boost)

if (NOT "$ENV{BOOST_DIR}" STREQUAL "") 
  set(BOOST_ROOT $ENV{BOOST_DIR})
endif ()

set(BOOST_LIBS system iostreams filesystem)
find_package(Boost 1.36.0 COMPONENTS ${BOOST_LIBS} REQUIRED)
if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS})
  set(RT_LINK ${Boost_LIBRARIES} ${RT_LINK})
endif()

if(USE_VTUNE)
  find_package(VTUNE)
  if (VTUNE_FOUND)
    add_definitions(-DGALOIS_USE_VTUNE)
    include_directories(${VTUNE_INCLUDE_DIRS})
    set(RT_LINK ${VTUNE_LIBRARIES} ${RT_LINK})
  else()
    message(WARNING "VTUNE required but not found")
  endif()
endif()

if(USE_PAPI)
  find_package(PAPI)
  if (PAPI_FOUND)
    add_definitions(-DGALOIS_USE_PAPI)
    include_directories(${PAPI_INCLUDE_DIR})
    set(RT_LINK ${PAPI_LIBRARIES} ${RT_LINK})
  else()
    message(WARNING "PAPI required but not found")
  endif()
endif()

if(USE_EXP)
  find_package(GMP) # for buchberger

  find_package(MPI REQUIRED) # for distributed
  set(CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS})
  set(CMAKE_CXX_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS})
  include_directories(${MPI_INCLUDE_PATH})
  set(RT_LINK ${MPI_LIBRARIES} ${RT_LINK})


  add_definitions(-DGALOIS_USE_EXP)
  include_directories("exp/include")

  if(USE_PROF) 
    add_definitions(-DGALOIS_USE_PROF)
  endif()

  find_package(TBB)
  if(TBB_FOUND)
    add_definitions(-DGALOIS_USE_TBB -DTBB_USE_THREADING_TOOLS=0)
    include_directories(${TBB_INCLUDE_DIRS})
    set(RT_LINK ${TBB_LIBRARIES} ${RT_LINK})
  else()
    # wget http://threadingbuildingblocks.org/uploads/77/173/3.0%20update%208/tbb30_20110704oss_lin.tgz
    message(WARNING "TBB specified but not found; have your sourced TBB_INSTALL_DIR/bin/tbbvars.sh?")
  endif()

  set(USE_SUBVERSION_REVISION on)
  if(USE_SUBVERSION_REVISION)
    include_directories("${PROJECT_BINARY_DIR}/include")
    include(GetSVNVersion)
    add_definitions(-DGALOIS_USE_SVNVERSION)
  endif()
  #find_package(MPI)
  #if(MPI_FOUND)
  #  add_definitions(-DGALOIS_MPI)
  #  include_directories(${MPI_INCLUDE_PATH})
  #  set(RT_LINK ${MPI_LIBRARIES} ${RT_LINK})
  #else()
  #  message(WARNING "MPI specified but not found")
  #endif()
endif()

# c++0x features
find_package(CXX0x)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CXX0X_FLAGS}")
if (HAS_CXX0X_RVALUE_REFERENCES)
  add_definitions(-DGALOIS_HAS_RVALUE_REFERENCES)
endif()

if(USE_GPROF)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pg")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
endif(USE_GPROF)

#solaris
if(CMAKE_SYSTEM MATCHES "SunOS.*")
  add_definitions(-DSOLARIS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64 -mcpu=niagara2 -lposix4")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -m64 -lposix4")
endif()

#Always include debug symbols
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g")

#Disable asserts
if(CMAKE_BUILD_TYPE MATCHES "Release")
  add_definitions("-DNDEBUG")
endif()

#more warnings
if(CMAKE_BUILD_TYPE MATCHES "Debug")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")
endif()

#always import c99 stdint functions into c++
#include(UseStdMacro) # HandleLLVMOptions.cmake already does this for us

include(CheckEndian)

set(LLVM_ENABLE_ASSERTIONS on) # enable assertions in non-release builds
include(llvm-extras)

###### Build Hacks ######

#XXX(ddn): Hack for lonestar machines
if(LIBNUMA_FOUND)
  if(LIBNUMA_OLD)
    add_definitions(-DGALOIS_USE_NUMA_OLD)
  endif()
endif()

#XXX(ddn): avoid broken includes on volta
#if(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
#  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CXX0X_FLAGS}")
#endif()

###### Global Functions ######
function(linkRuntime name) 
  target_link_libraries(${name} ${ARGN})
  target_link_libraries(${name} galois ${RT_LINK})
  if(USE_SUBVERSION_REVISION)
    add_dependencies(${name} svnversion)
  endif()
endfunction(linkRuntime)

include(ParseArguments)

function(compileApp name) 
  if(ARGN)
    set(Sources ${ARGN})
  else()
    file(GLOB Sources *.cpp)
  endif()
  add_executable(${name} ${Sources})
endfunction (compileApp)

function(app name)
  PARSE_ARGUMENTS(APP "REQUIRES;EXTLIBS" "" ${ARGN})

  foreach(required ${APP_REQUIRES})
    if(${${required}})

    else()
      message(WARNING "NOT compiling ${name} (missing: ${required})")
      return()
    endif()
  endforeach()

  compileApp(${name} ${APP_DEFAULT_ARGS})
  linkRuntime(${name} ${APP_EXTLIBS})
  install(TARGETS ${name} DESTINATION bin)
endfunction(app)

###### Source finding ######

configure_file("${PROJECT_SOURCE_DIR}/include/Galois/config.h.in" "${PROJECT_BINARY_DIR}/include/Galois/config.h")
include_directories("include")

add_subdirectory(lib)
add_subdirectory(src)
add_subdirectory(tools)
add_subdirectory(inputs)
add_subdirectory(test)
add_subdirectory(scripts)
add_subdirectory(apps)
if(USE_EXP)
  add_subdirectory(exp)
endif()

###### Documentation ######
include(UseDoxygen OPTIONAL)

###### Distribution ######
include(InstallRequiredSystemLibraries)
set(CPACK_GENERATOR "TGZ")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/COPYRIGHT")
set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README")
set(CPACK_PACKAGE_VERSION_MAJOR ${GALOIS_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${GALOIS_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${GALOIS_VERSION_PATCH})
include(CPack)

###### Installation ######
install(DIRECTORY include/ DESTINATION include PATTERN ".svn" EXCLUDE PATTERN ".in" EXCLUDE)

