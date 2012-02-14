cmake_minimum_required(VERSION 2.6)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")
project(Galois)
set(GALOIS_VERSION_MAJOR "2")
set(GALOIS_VERSION_MINOR "1")
set(GALOIS_VERSION_PATCH "3")
set(GALOIS_COPYRIGHT_YEAR "2012") # Also in COPYRIGHT
site_name(SITE_NAME)

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "No build type selected, default to Release")
  set(CMAKE_BUILD_TYPE "Release")
endif()

###### Options (alternatively pass as options to cmake -DName=Value) ######

#use profiling
#set(USE_PROFILING ON)

#use alternate compiler
#set(USE_ICC ON)
#set(USE_CLANG ON)

#Enable iss specific options; should be OFF in the general release
#use experimental features
set(USE_EXP ON) 

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
    add_definitions("-wd981 -wd383 -wd869 -wd2196 -wd279 -std=c++0x")
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
find_library(LIBNUMA numa)
set(USE_LIBNUMA TRUE CACHE BOOL "Use libnuma to control memory allocation.")
if(USE_LIBNUMA)
  if (LIBNUMA)
    message(STATUS "NUMA in use")
    set(RT_LINK numa ${RT_LINK})
    add_definitions(-DGALOIS_NUMA)
  else()
    message(STATUS "NUMA specified but libnuma not found")
  endif()
endif()

#LAPACK
enable_language(Fortran)
find_package(LAPACK)

#Boost
#not needed now but will be needed for newer version of Boost after 1.4X ish
set(Boost_ADDITIONAL_VERSIONS "1.40" "1.40.0" "1.47" "1.47.0")
set(Boost_USE_MULTITHREADED OFF) # newer boost builds don't use -mt extension
#Other options that can be used with find_package(Boost)
# set(BOOST_ROOT /path/)
# set(Boost_USE_STATIC_LIBS ON)
# find_package(Boost 1.34.0 REQUIRED COMPONENTS system serialization graph_parallel mpi)
# set(BOOST_LIBS system)
if(${SITE_NAME} MATCHES "delaunay.*")
  find_package(Boost 1.34.0 COMPONENTS ${BOOST_LIBS})
else()
  find_package(Boost 1.34.0 COMPONENTS ${BOOST_LIBS} REQUIRED)
endif()
include_directories(${Boost_INCLUDE_DIRS})
set(RT_LINK ${Boost_LIBRARIES} ${RT_LINK})

if(USE_EXP)
  set(VTUNE_INCLUDE "/opt/intel/vtune_amplifier_xe_2011/include")
  set(VTUNE_LIB "/opt/intel/vtune_amplifier_xe_2011/lib64/libittnotify.a")
  if(EXISTS ${VTUNE_INCLUDE})
    add_definitions(-DGALOIS_VTUNE)
    include_directories(${VTUNE_INCLUDE})
    set(RT_LINK ${VTUNE_LIB} dl ${RT_LINK})
  endif()

  find_package(OpenMP)
  if(OPENMP_FOUND)
    set(RT_LINK -lgomp ${RT_LINK})
  endif()

  find_package(TBB)
  if(TBB_FOUND)
    add_definitions(-DGALOIS_TBB -DTBB_USE_THREADING_TOOLS=0)
    include_directories(${TBB_INCLUDE_DIRS})
    set(RT_LINK ${TBB_LIBRARIES} ${RT_LINK})
  else()
    # wget http://threadingbuildingblocks.org/uploads/77/173/3.0%20update%208/tbb30_20110704oss_lin.tgz
    message(STATUS "Could NOT find TBB; have your sourced TBB_INSTALL_DIR/bin/tbbvars.sh?")
  endif()

  find_package(MPI)
  if(MPI_FOUND)
    add_definitions(-DGALOIS_MPI)
    include_directories(${MPI_INCLUDE_PATH})
    set(RT_LINK ${MPI_LIBRARIES} ${RT_LINK})
  endif()
endif()

# c++0x features

# enable -std=c++0x if possible
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-std=c++0x HAS_CXX0X)
if(HAS_CXX0X)
#  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
endif()

include(CheckCXX0xFeatures)

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
add_definitions(-D__STDC_LIMIT_MACROS)
add_definitions(-D__STDC_CONSTANT_MACROS)

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
    if(NOT ${${required}})
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
configure_file("${PROJECT_SOURCE_DIR}/scripts/make_dist.sh.in" "${PROJECT_BINARY_DIR}/scripts/make_dist.sh")

add_subdirectory(lib)
add_subdirectory(src)
add_subdirectory(apps)
add_subdirectory(tools)
add_subdirectory(inputs)
add_subdirectory(test)

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
#install(DIRECTORY scripts/ DESTINATION scripts PATTERN ".svn" EXCLUDE PATTERN ".in" EXCLUDE)
execute_process(COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/scripts ${CMAKE_BINARY_DIR}/scripts)
