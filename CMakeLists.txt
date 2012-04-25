cmake_minimum_required(VERSION 2.6)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")
project(Galois)
set(GALOIS_VERSION_MAJOR "2")
set(GALOIS_VERSION_MINOR "1")
set(GALOIS_VERSION_PATCH "4")
set(GALOIS_COPYRIGHT_YEAR "2012") # Also in COPYRIGHT
site_name(SITE_NAME)

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "No build type selected, default to Release")
  set(CMAKE_BUILD_TYPE "Release")
endif()

###### Options (alternatively pass as options to cmake -DName=Value) ######

#use profiling
set(USE_PROFILING OFF CACHE BOOL "Enable GCC profiling")

#use alternate compiler
set(USE_ICC OFF CACHE BOOL "Use intel compiler")
set(USE_CLANG OFF CACHE BOOL "Use clang compiler")

#Enable iss specific options; should be OFF in the general release
set(USE_EXP OFF CACHE BOOL "Use experimental features") 
set(USE_VTUNE ON CACHE BOOL "Use vtune (if found)")
set(USE_DET OFF CACHE BOOL "Use deterministic execution")

###### Optional programs and default paths ######

#find better c++ compilers
find_program(CLANGXX clang++)
find_program(CXX_4.5 g++-4.5)
find_program(ICCXX icpc)

#always use newer gcc
if(NOT ${CXX_4.5} STREQUAL "CXX_4.5-NOTFOUND")
  set(CMAKE_CXX_COMPILER ${CXX_4.5})
endif()

if(NOT ${CLANGXX} STREQUAL "CLANGXX-NOTFOUND")
  if(USE_CLANG)
    set(CMAKE_CXX_COMPILER ${CLANGXX})  
  endif()
endif()

if(NOT ${ICCXX} STREQUAL "ICCXX-NOTFOUND")
  if(USE_ICC)
    set(CMAKE_CXX_COMPILER ${ICCXX})
    add_definitions("-wd68 -wd981 -wd383 -wd869 -wd2196 -wd279 -std=c++0x")
  endif()
endif()

if(CMAKE_SYSTEM MATCHES "SunOS.*")
  set(USE_SOLARIS ON)
endif()

###### Configure (users don't need to go beyond here) ######
enable_testing()

#PThreads
find_package(Threads)
set(RT_LINK ${CMAKE_THREAD_LIBS_INIT} ${RT_LINK})

#NUMA (linux)
find_package(LIBNUMA)
if(LIBNUMA_FOUND)
  set(RT_LINK numa ${RT_LINK})
  if(LIBNUMA_OLD)
    add_definitions(-DGALOIS_NUMA_OLD)
  endif()
  add_definitions(-DGALOIS_NUMA)
endif()

#LAPACK
enable_language(Fortran)
find_package(LAPACK)

#CILK
include(CheckCilk)

# ZLIB
find_package(ZLIB)
if(ZLIB_FOUND)
  include_directories(${ZLIB_INCLUDE_DIRS})
  set(RT_LINK ${ZLIB_LIBRARIES} ${RT_LINK})
endif(ZLIB_FOUND)

#Boost
#not needed now but will be needed for newer version of Boost after 1.4X ish
set(Boost_ADDITIONAL_VERSIONS "1.40" "1.40.0" "1.47" "1.47.0")
set(Boost_USE_MULTITHREADED OFF) # newer boost builds don't use -mt extension
#Other options that can be used with find_package(Boost)
# set(BOOST_ROOT /path/)
# set(Boost_NO_SYSTEM_PATHS TRUE) # useful with BOOST_ROOT
set(BOOST_LIBS system iostreams filesystem)
find_package(Boost 1.36.0 COMPONENTS ${BOOST_LIBS} REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
set(RT_LINK ${Boost_LIBRARIES} ${RT_LINK})

if(USE_EXP)
  add_definitions(-DGALOIS_EXP)
  include_directories("exp/include")

  if(USE_DET)
    add_definitions(-DGALOIS_DET)
  endif()

  if(USE_VTUNE)
    find_package(VTUNE)
    if (VTUNE_FOUND)
      add_definitions(-DGALOIS_VTUNE)
      include_directories(${VTUNE_INCLUDE_DIRS})
      set(RT_LINK ${VTUNE_LIBRARIES} ${RT_LINK})
    else()
      message(WARNING "VTUNE required but not found")
    endif()
  endif()

  find_package(TBB)
  if(TBB_FOUND)
    add_definitions(-DGALOIS_TBB -DTBB_USE_THREADING_TOOLS=0)
    include_directories(${TBB_INCLUDE_DIRS})
    set(RT_LINK ${TBB_LIBRARIES} ${RT_LINK})
  else()
    # wget http://threadingbuildingblocks.org/uploads/77/173/3.0%20update%208/tbb30_20110704oss_lin.tgz
    message(WARNING "TBB specified but not found; have your sourced TBB_INSTALL_DIR/bin/tbbvars.sh?")
  endif()

  find_package(MPI)
  if(MPI_FOUND)
    add_definitions(-DGALOIS_MPI)
    include_directories(${MPI_INCLUDE_PATH})
    set(RT_LINK ${MPI_LIBRARIES} ${RT_LINK})
  else()
    message(WARNING "MPI specified but not found")
  endif()
endif()

# c++0x features
find_package(CXX0x)

if(USE_PROFILING)
  add_definitions(-pg)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
endif(USE_PROFILING)

#solaris
if(USE_SOLARIS)
  add_definitions(-DSOLARIS)
  add_definitions("-m64")
  add_definitions("-mcpu=niagara2")
  add_definitions("-lposix4")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -m64 -lposix4")
endif()

#Always include debug symbols
# add_definitions("-g -fno-inline")
add_definitions("-g")

#Disable asserts
if(CMAKE_BUILD_TYPE MATCHES "Release")
  add_definitions("-DNDEBUG")
endif()

#more warnings
if(CMAKE_BUILD_TYPE MATCHES "Debug")
  add_definitions("-Wall")
endif()

#always import c99 stdint functions into c++
include(UseStdMacro)

include(CheckEndian)

###### Global Functions ######
function(linkRuntime name) 
  target_link_libraries(${name} ${ARGN})
  target_link_libraries(${name} galois ${RT_LINK})
  if(USE_EXP)
    target_link_libraries(${name} galois_exp ${RT_LINK})
  endif()
endfunction (linkRuntime)

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
set(LLVM_ENABLE_ASSERTIONS on) # enable assertions in non-release builds
include(llvm-extras)

configure_file("${PROJECT_SOURCE_DIR}/include/config.h.in" "${PROJECT_BINARY_DIR}/include/config.h")
include_directories("include" "${PROJECT_BINARY_DIR}/include")

add_subdirectory(lib)
add_subdirectory(src)
add_subdirectory(apps)
add_subdirectory(tools)
add_subdirectory(inputs)
add_subdirectory(test)
add_subdirectory(scripts)
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

