cmake_minimum_required(VERSION 2.6)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")
project(Galois)
set(GALOIS_VERSION_MAJOR "2")
set(GALOIS_VERSION_MINOR "2")
set(GALOIS_VERSION_PATCH "0")
set(GALOIS_COPYRIGHT_YEAR "2013") # Also in COPYRIGHT

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "No build type selected, default to Release")
  set(CMAKE_BUILD_TYPE "Release")
endif()

###### Options (alternatively pass as options to cmake -DName=Value) ######

#use profiling
set(USE_GPROF OFF CACHE BOOL "Enable GCC profiling")
set(USE_VTUNE ON CACHE BOOL "Use vtune (if found) for profiling")
set(USE_PAPI OFF CACHE BOOL "Use PAPI counters for profiling")
set(USE_STRICT_CONFIG OFF CACHE BOOL "Instead of falling back gracefully, fail")

#Enable iss specific options; should be OFF in the general release; all guarded by USE_EXP
set(USE_EXP OFF CACHE BOOL "Use experimental features") 
set(EXP_DOALL "PTHREAD" CACHE STRING "Which type of implementation of parallel_doall")
set(USE_PROF OFF CACHE BOOL "Use profiling specific features")
set(USE_SUBVERSION_REVISION ON CACHE BOOL "Embed subversion numbers")

###### Configure (users don't need to go beyond here) ######
enable_testing()

#ICC
if(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -wd68 -wd981 -wd383 -wd869 -wd2196 -wd279 -wd2504 -wd2943")
endif()

#PThreads
find_package(Threads REQUIRED)

#NUMA (linux)
find_package(LIBNUMA)
if(LIBNUMA_FOUND)
  add_definitions(-DGALOIS_USE_NUMA)
elseif(USE_STRICT_CONFIG)
  message(FATAL_ERROR "Need libnuma")
endif()

#LAPACK
enable_language(Fortran)
find_package(LAPACK)

#CILK
include(CheckCilk)

#HugePages
include(CheckHugePages)
if(NOT HAVE_HUGEPAGES AND USE_STRICT_CONFIG)
  message(FATAL_ERROR "Need huge pages")
endif()

#ZLIB
find_package(ZLIB REQUIRED)
include_directories(${ZLIB_INCLUDE_DIRS})

#Boost
#not needed now but will be needed for newer version of Boost after 1.4X ish
set(Boost_ADDITIONAL_VERSIONS "1.40" "1.40.0" "1.47" "1.47.0" "1.49" "1.49.0" "1.51.0")
set(Boost_USE_MULTITHREADED OFF) # newer boost builds don't use -mt extension
#Other options that can be used with find_package(Boost)

if (NOT "$ENV{BOOST_DIR}" STREQUAL "") 
  set(BOOST_ROOT $ENV{BOOST_DIR})
endif ()

set(BOOST_LIBS system iostreams filesystem)
#set(BOOST_LIBS)
find_package(Boost 1.36.0 COMPONENTS ${BOOST_LIBS} REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

if(USE_VTUNE)
  find_package(VTUNE)
  if (VTUNE_FOUND)
    add_definitions(-DGALOIS_USE_VTUNE)
    include_directories(${VTUNE_INCLUDE_DIRS})
  else()
    message(WARNING "VTUNE required but not found")
  endif()
endif()

if(USE_PAPI)
  find_package(PAPI)
  if (PAPI_FOUND)
    add_definitions(-DGALOIS_USE_PAPI)
    include_directories(${PAPI_INCLUDE_DIR})
  else()
    message(WARNING "PAPI required but not found")
  endif()
endif()

#check for incompatible GCC
if (CMAKE_COMPILER_IS_GNUCC)
execute_process(COMMAND ${CMAKE_C_COMPILER} -dumpversion
                OUTPUT_VARIABLE GCC_VERSION)
if (GCC_VERSION VERSION_GREATER 4.6 OR GCC_VERSION VERSION_EQUAL 4.6)
        message(STATUS "Gcc Version >= 4.6")
else()
	message(FATAL_ERROR "GCC must be 4.6 or higher")
endif()
endif()

if(USE_EXP)
  add_definitions(-DGALOIS_USE_EXP)
  include_directories("exp/include")

  if(USE_PROF) 
    add_definitions(-DGALOIS_USE_PROF)
  endif()

  find_package(TBB)
  if(TBB_FOUND)
    add_definitions(-DGALOIS_USE_TBB -DTBB_USE_THREADING_TOOLS=0)
    include_directories(${TBB_INCLUDE_DIRS})
  endif()

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

# c++11 features
find_package(CXX11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CXX11_FLAGS}")

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

#Enable architecture-specific optimizations
if(CMAKE_BUILD_TYPE MATCHES "Release" AND CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
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

###### Global Functions ######

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
    if(${${required}} MATCHES "TRUE")

    else()
      message(WARNING "NOT compiling ${name} (missing: ${required})")
      return()
    endif()
  endforeach()

  compileApp(${name} ${APP_DEFAULT_ARGS})
  target_link_libraries(${name} ${APP_EXTLIBS})
  target_link_libraries(${name} galois)
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
set(DOXYFILE_SOURCE_DIR "src\" \"include")
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

