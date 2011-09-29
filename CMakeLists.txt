cmake_minimum_required(VERSION 2.6)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")
project(Galois)
site_name(SITE_NAME)

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "No build type selected, default to Release")
  set(CMAKE_BUILD_TYPE "Release")
endif()

###### Options (alternatively pass as options to cmake -DName=Value ######

#use pthreads
set(USE_PTHREAD ON)

#compile for cray xmt
#set(USE_CRAY ON)

#use icc (should pick right path for icc)
#set(USE_ICC ON)
#set(ICC_COMPILER /opt/intel/Compiler/11.1/072/bin/intel64/icpc)

#use profiling
#set(USE_PROFILING ON)

# Enable iss specific options; should be OFF in the general release
set(USE_ISS ON)

##### Machine-specific settings (probably not useful to you) ######
if(SITE_NAME MATCHES "volta.*")
  set(USE_NUMA ON)
endif()

find_program(CXX_4.5 g++-4.5)
if(NOT ${CXX_4.5} STREQUAL "CXX_4.5-NOTFOUND")
  set(CMAKE_CXX_COMPILER ${CXX_4.5})
endif()

#if(SITE_NAME MATCHES "delaunay.*")
#  set(CMAKE_CXX_COMPILER /opt/csw/gcc4/bin/g++)
#  set(CMAKE_CC_COMPILER /opt/csw/gcc4/bin/gcc)
#  # Override broken include files in
#  #  /opt/csw/gcc4/lib/gcc/sparc-sun-solaris2.8/4.5.1/include-fixed/
#  add_definitions(-nostdinc)
#  add_definitions(-I/usr/include)
#  add_definitions(-I/opt/csw/gcc4/include/c++/4.5.1)
#  add_definitions(-I/opt/csw/gcc4/include/c++/4.5.1/sparc-sun-solaris2.8)
#endif()

if(CMAKE_SYSTEM MATCHES "SunOS.*")
  set(USE_SOLARIS ON)
endif()

###### Configure (users don't need to go beyond here) ######
enable_testing()

#LAPACK
enable_language(Fortran)
find_package(LAPACK)

#Boost
# not needed now but will be needed for newer version of Boost after 1.4X ish
set(Boost_ADDITIONAL_VERSIONS "1.40" "1.40.0" "1.47" "1.47.0")
#set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_MULTITHREADED OFF) # newer boost builds don't use -mt extension
#set(BOOST_ROOT /net/faraday/workspace/ddn/local)
#find_package(Boost 1.34.0 REQUIRED COMPONENTS system serialization graph_parallel mpi)
#set(BOOST_LIBS system)
if(SITE_NAME MATCHES "delaunay.*")
  find_package(Boost 1.34.0 COMPONENTS ${BOOST_LIBS})
else()
  find_package(Boost 1.34.0 COMPONENTS ${BOOST_LIBS} REQUIRED)
endif()
include_directories(${Boost_INCLUDE_DIRS})

#PThreads
if(USE_PTHREAD)
  find_package(Threads)
  add_definitions(-DGALOIS_PTHREAD)
endif(USE_PTHREAD)

find_package(MPI)
if(MPI_FOUND)
  add_definitions(-DGALOIS_MPI)
  include_directories(${MPI_INCLUDE_PATH})
endif()

if(USE_NUMA)
  add_definitions(-DGALOIS_NUMA)
endif(USE_NUMA)

#vtune
set(VTUNE_INCLUDE "/opt/intel/vtune_amplifier_xe_2011/include")
set(VTUNE_LIB "/opt/intel/vtune_amplifier_xe_2011/lib64/libittnotify.a")
if(EXISTS ${VTUNE_INCLUDE})
  set(USE_VTUNE ON)
  add_definitions(-DGALOIS_VTUNE)
  include_directories(${VTUNE_INCLUDE})
endif()

# c++0x features

# enable -std=c++0x if possible
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-std=c++0x HAS_CXX0X)
if(HAS_CXX0X)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
endif()

include(CheckCXX0xFeatures)

#tbb
find_package(TBB)

if(TBB_FOUND)
  add_definitions(-DGALOIS_TBB)  
  include_directories(${TBB_INCLUDE_DIRS})
else()
  # wget http://threadingbuildingblocks.org/uploads/77/173/3.0%20update%208/tbb30_20110704oss_lin.tgz
  message(STATUS "Could NOT find TBB; have your sourced TBB_INSTALL_DIR/bin/tbbvars.sh?")
endif()

#cray
if(USE_CRAY)
  add_definitions(-DGALOIS_CRAY)
endif(USE_CRAY)

if(USE_PROFILING)
  add_definitions(-pg)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
endif(USE_PROFILING)

if(USE_ISS)
  add_definitions(-DGALOIS_ISS)
endif()

#solaris
if(USE_SOLARIS)
  add_definitions(-DSOLARIS)
  add_definitions("-m64")
  add_definitions("-mcpu=niagara2")
  add_definitions("-lposix4")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -m64 -lposix4")
endif()

if(USE_ICC)
  set(CMAKE_CXX_COMPILER ${ICC_COMPILER})
  #These help the intel compiler
  add_definitions("-wd981 -wd383 -wd869 -wd2196 -wd279")
endif(USE_ICC)

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

###### Global Functions ######
function(linkRuntime name) 
  target_link_libraries(${name} galois) 
  target_link_libraries(${name} ${CMAKE_THREAD_LIBS_INIT})
  if(NOT USE_SOLARIS)
    target_link_libraries(${name} hoard)
  endif()

  if(USE_VTUNE)
    target_link_libraries(${name} ${VTUNE_LIB})
    target_link_libraries(${name} dl)
  endif()

  if(USE_NUMA)
    target_link_libraries(${name} numa)
  endif()

  if(TBB_FOUND)
    target_link_libraries(${name} ${TBB_LIBRARIES})
  endif()

  if(MPI_FOUND)
    target_link_libraries(${name} ${MPI_LIBRARIES})
  endif()
  
  target_link_libraries(${name} ${Boost_LIBRARIES})
endfunction (linkRuntime)

###### Source finding ######
include_directories("include")

add_subdirectory(lib)
add_subdirectory(src)
add_subdirectory(apps)
add_subdirectory(tools)
add_subdirectory(inputs)
add_subdirectory(test)

###### Documentation ######
include(UseDoxygen OPTIONAL)

###### Distribution ######
include(InstallRequiredSystemLibraries)
set(CPACK_GENERATOR "TGZ")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/COPYRIGHT")
set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README")
set(CPACK_PACKAGE_VERSION_MAJOR "2")
set(CPACK_PACKAGE_VERSION_MINOR "1")
set(CPACK_PACKAGE_VERSION_PATCH "0")
include(CPack)

###### Installation ######
install(DIRECTORY include/ DESTINATION include PATTERN ".svn" EXCLUDE)
