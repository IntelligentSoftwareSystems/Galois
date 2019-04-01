cmake_minimum_required(VERSION 3.2.0)

include(GNUInstallDirs)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")
project(Galois)
set(GALOIS_VERSION_MAJOR "4")
set(GALOIS_VERSION_MINOR "0")
set(GALOIS_VERSION_PATCH "0")
set(GALOIS_VERSION ${GALOIS_VERSION_MAJOR}.${GALOIS_VERSION_MINOR}.${GALOIS_VERSION_PATCH})
set(GALOIS_COPYRIGHT_YEAR "2018") # Also in COPYRIGHT

#adding additional compiler flags for mpip profiling
#set(LIBUNWIND_DIR "/net/ohm/export/cdgc/gill/modules/usr/local/lib")
#set(MPIP_DIR "/net/ohm/export/cdgc/gill/modules/mpip/lib")
#set(GCC_MPIP_LINK_FLAGS "-L${MPIP_DIR} -L${LIBUNWIND_DIR} -lmpiP -lbfd -liberty -lm -lunwind")
#link_directories(LIBUNWIND_DIR MPIP_DIR)


if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "No build type selected, default to release")
  # cmake default flags with relwithdebinfo is -O2 -g
  # cmake default flags with release is -O3 -DNDEBUG
  set(CMAKE_BUILD_TYPE "Release")
endif()

###### Options (alternatively pass as options to cmake -DName=Value) ######
###### Distributed-heterogeneous features ######
set(ENABLE_DIST_GALOIS OFF CACHE BOOL "Enable distributed features")
set(ENABLE_HETERO_GALOIS OFF CACHE BOOL "Enable heterogeneous features")
set(USE_LCI OFF CACHE BOOL "Use LCI network runtime instead of MPI")
set(USE_BARE_MPI OFF CACHE BOOL "Use MPI directly (no dedicated network-runtime thread)")
set(CUDA_CAPABILITY "3.7,6.1" CACHE STRING "Comma-separated CUDA capability version numbers")
set(REPORT_COMM_STATS OFF CACHE BOOL "Report more detailed statistics of communication")
set(REPORT_PER_ROUND_STATS OFF CACHE BOOL "Report statistics of each round of execution")
###### General features ######
set(USE_GPROF OFF CACHE BOOL "Enable GCC profiling")
set(USE_VTUNE OFF CACHE BOOL "Use VTune for profiling")
set(USE_PAPI OFF CACHE BOOL "Use PAPI counters for profiling")
set(USE_HPCTK OFF CACHE BOOL "Use HPCToolKit for profiling")
set(USE_STRICT_CONFIG OFF CACHE BOOL "Instead of falling back gracefully, fail")
set(USE_LONGJMP_ABORT ON CACHE BOOL "Use longjmp instead of exceptions to signal aborts")
set(USE_SANITIZER OFF CACHE BOOL "Use address and memory sanatizer")
set(INSTALL_APPS OFF CACHE BOOL "Install apps as well as library")
set(SKIP_COMPILE_APPS OFF CACHE BOOL "Skip compilation of applications using Galois library")
set(GRAPH_LOCATION "" CACHE PATH "Location of inputs for tests if downloaded/stored separately.")

if(WIN32 AND NOT CYGWIN)
  set(DEFAULT_INSTALL_CMAKE_DIR "${CMAKE_INSTALL_PREFIX}/CMake")
else()
  set(DEFAULT_INSTALL_CMAKE_DIR "${CMAKE_INSTALL_LIBDIR}/CMake/Galois")
endif()

# This option is automatically handled by CMake.
# It makes add_library build a shared lib unless STATIC is explicitly specified.
# Putting this here is mostly just a placeholder so people know it's an option.
# Currently this is really only intended to change anything for the libgalois_shmem target.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libraries.")

set(INSTALL_CMAKE_DIR "${DEFAULT_INSTALL_CMAKE_DIR}" CACHE PATH "Installation directory for CMake files")
# Make relative path absolute
if(NOT IS_ABSOLUTE "${INSTALL_CMAKE_DIR}")
  set(INSTALL_CMAKE_DIR "${CMAKE_INSTALL_PREFIX}/${INSTALL_CMAKE_DIR}")
endif()

file(RELATIVE_PATH RELATIVE_INCLUDE_FROM_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}" "${CMAKE_INSTALL_FULL_INCLUDEDIR}")
file(RELATIVE_PATH RELATIVE_LIB_FROM_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}" "${CMAKE_INSTALL_FULL_LIBDIR}")
file(RELATIVE_PATH RELATIVE_BIN_FROM_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}" "${CMAKE_INSTALL_FULL_BINDIR}")

# Enable iss specific options; should be OFF in the general release; all guarded by USE_EXP
set(USE_EXP OFF CACHE BOOL "Use experimental features")
set(USE_DIST_EXP OFF CACHE BOOL "Use experimental features in distributed branch")
set(USE_HTM OFF CACHE BOOL "Use HTM")
set(USE_PROF OFF CACHE BOOL "Use profiling specific features")
set(USE_EMBEDDED_REVISION ON CACHE BOOL "Embed revision numbers")

# Enable cross compilation for specific architectures: For eg. -march=knl for knl architecture
set(USE_ARCH native CACHE STRING "Use specific architecture for cross compilation (Default : native)")

###### Configure (users don't need to go beyond here) ######

enable_testing()

###### Configure compiler ######

#Always include debug info and aggressive warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -Wall")

# generate compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT "$ENV{GCC_BIN}" STREQUAL "")
  # Preserve captured environment variable during reconfiguration
  set(CXX_TOOLCHAIN "$ENV{GCC_BIN}" CACHE STRING "GCC Toolchain")
endif()

# ICC
if(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
  execute_process(COMMAND ${CMAKE_CXX_COMPILER} -dumpversion
                  OUTPUT_VARIABLE ICC_VERSION)
  if(ICC_VERSION VERSION_LESS 16.0)
    message(FATAL_ERROR "ICC must be 16.0 or higher; found: ${ICC_VERSION} instead")
  endif()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -wd68 -wd981 -wd383 -wd869 -wd2196 -wd279 -wd2504 -wd2943 -wd32013 -wd3373")

  if(CXX_TOOLCHAIN)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -gxx-name=${CXX_TOOLCHAIN}/g++")
  else()
    message(STATUS "Using default GCC toolchain; set environment variable GCC_BIN to override")
  endif()
endif()

# Clang
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  if(CXX_TOOLCHAIN)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -gcc-toolchain ${CXX_TOOLCHAIN}/..")
  else()
    message(STATUS "Using default GCC toolchain; set environment variable GCC_BIN to override")
  endif()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fcolor-diagnostics")
endif()

if(CYGWIN)
  message(STATUS "On Cygwin")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -U__STRICT_ANSI__")
endif()

# check for incompatible GCC
if(CMAKE_COMPILER_IS_GNUCC)
  execute_process(COMMAND ${CMAKE_CXX_COMPILER} -dumpversion
                  OUTPUT_VARIABLE GCC_VERSION)
  if(GCC_VERSION VERSION_LESS 5.2)
    message(FATAL_ERROR "GCC must be 5.2 or higher")
  endif()
  #new debugging optimization level
  # if(CMAKE_BUILD_TYPE MATCHES "Debug")
    # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Og")
    # set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Og")
  # endif()
endif()

# Enable architecture-specific optimizations
if(NOT ENABLE_HETERO_GALOIS)
  find_package(ArchFlags)
  if(ARCH_FLAGS_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ARCH_CXX_FLAGS}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${ARCH_C_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${ARCH_EXE_FLAGS}")
  endif()
endif()

# GNU profiling
if(USE_GPROF)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pg")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
endif(USE_GPROF)

if(USE_SANITIZER)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer -fno-optimize-sibling-calls")
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) #...without compiler extensions like gnu++11

###### Configure features ######

add_definitions(-DGALOIS_VERSION_MAJOR=${GALOIS_VERSION_MAJOR})
add_definitions(-DGALOIS_VERSION_MINOR=${GALOIS_VERSION_MINOR})
add_definitions(-DGALOIS_VERSION_PATCH=${GALOIS_VERSION_PATCH})
add_definitions(-DGALOIS_VERSION=${GALOIS_VERSION_MAJOR}.${GALOIS_VERSION_MINOR}.${GALOIS_VERSION_PATCH})
add_definitions(-DGALOIS_COPYRIGHT_YEAR=${GALOIS_COPYRIGHT_YEAR})


# Distributed-heterogeneous features
if(ENABLE_HETERO_GALOIS)
  set(ENABLE_DIST_GALOIS ON)
endif()
if(ENABLE_DIST_GALOIS)
  find_package(MPI REQUIRED) # required even if USE_LCI
  set(CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS})
  set(CMAKE_CXX_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS})
  include_directories(${MPI_INCLUDE_PATH})

  if (USE_LCI)
    add_definitions(-DGALOIS_USE_LWCI)
    set(LWCI_ROOT "${CMAKE_SOURCE_DIR}/liblci")
    set(LWCI_INCLUDE "${LWCI_ROOT}/include")
    set(LWCI_LIBRARY "${LWCI_ROOT}/liblci.a")
    add_custom_command(
      OUTPUT ${LWCI_LIBRARY}
      COMMAND make -C ${LWCI_ROOT}
    )
    add_custom_target(lci DEPENDS ${LWCI_LIBRARY})
  endif(USE_LCI)

  if(ENABLE_HETERO_GALOIS)
    SET(CUDA_SEPARABLE_COMPILATION ON)
    find_package(CUDA REQUIRED)
    set(CUDA_PROPAGATE_HOST_FLAGS off)
    set(CUDA_SEPARABLE_COMPILATION on)
    set(CUDA_HOST_COMPILER g++)

    string(REPLACE "." "" GENCODES ${CUDA_CAPABILITY})
    string(REPLACE "," ";" GENCODES ${GENCODES})
    foreach(GENCODE ${GENCODES})
      set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_${GENCODE},code=sm_${GENCODE})
    endforeach()

    cuda_include_directories("${CMAKE_SOURCE_DIR}/libgpu/include")

    # MGPU v1.1
    set(MGPU_ROOT "${CMAKE_SOURCE_DIR}/libgpu/moderngpu") # only required headers
    cuda_include_directories("${MGPU_ROOT}/include")

    # CUB v1.6.4
    set(CUB_ROOT "${CMAKE_SOURCE_DIR}/libgpu/cub") # only required headers
    cuda_include_directories("${CUB_ROOT}")

    #find_package(OpenCL REQUIRED)
  endif(ENABLE_HETERO_GALOIS)
endif(ENABLE_DIST_GALOIS)

# Experimental features
if(USE_EXP)
  set(USE_VTUNE ON)
  set(USE_PAPI ON)
  add_definitions(-DGALOIS_USE_EXP)
  include_directories("libgalois/experimental/include")
  if(ENABLE_DIST_GALOIS)
    include_directories("libdist/experimental/include")
  endif()

  find_package(OpenMP)
  if(OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  endif()

  find_package(TBB)
  if(TBB_FOUND)
    include_directories(${TBB_INCLUDE_DIRS})
  endif()
  
  find_package(CBLAS)
  if(CBLAS_FOUND)
    include_directories(${BLAS_INCLUDE_DIRS})
  endif()

  if(USE_PROF)
    add_definitions(-DGALOIS_USE_PROF)
  endif()

  if(USE_EMBEDDED_REVISION)
    include(GetGitVersion)
    set(GALOIS_USE_EMBEDDED_REVISION on)
  endif()

  if(USE_HTM)
    if(CMAKE_CXX_COMPILER_ID MATCHES "XL")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qtm -qsmp=speculative")
      set(GALOIS_USE_HTM on)
      set(GALOIS_USE_SEQ_ONLY on)
      set(USE_LONGJMP_ABORT on)
    else()
      message(FATAL_ERROR "Hardware transactional memory not supported")
    endif()
  endif()

endif()

if(USE_VTUNE)
  set(VTune_ROOT /opt/intel/vtune_amplifier)
  find_package(VTune)
  message(STATUS "VTUNE: ${VTune_INCLUDE_DIRS}")

  if (VTune_FOUND)
    include_directories(${VTune_INCLUDE_DIRS})
    add_definitions(-DGALOIS_USE_VTUNE)
  else()
    message(WARNING "VTune not found")
  endif()
endif()

if(USE_PAPI)
  if (PAPI_ROOT STREQUAL "")
    set(PAPI_ROOT /usr)
  endif()
  find_package(PAPI)
  message(STATUS "PAPI: ${PAPI_INCLUDE_DIRS}")

  if (PAPI_FOUND)
    include_directories(${PAPI_INCLUDE_DIRS})
    add_definitions(-DGALOIS_USE_PAPI)
  else()
    message(WARNING "PAPI not found")
  endif()
endif()

# PThreads
find_package(Threads REQUIRED)

#llvm ADT and command line parser
#find_package(LLVM)

# include(CheckCilk)
include(CheckMmap)

# HugePages
include(CheckHugePages)
if(NOT HAVE_HUGEPAGES AND USE_STRICT_CONFIG)
  message(FATAL_ERROR "Need huge pages")
endif()

# Longjmp
if(USE_LONGJMP_ABORT)
  add_definitions(-DGALOIS_USE_LONGJMP_ABORT)
endif()

# Boost
set(Boost_ADDITIONAL_VERSIONS "1.58.0" "1.60.0" "1.61.0" "1.62.0")
set(Boost_USE_MULTITHREADED OFF)
if(NOT "$ENV{BOOST_DIR}" STREQUAL "")
  set(BOOST_ROOT $ENV{BOOST_DIR})
endif()
find_package(Boost 1.58.0 REQUIRED COMPONENTS serialization iostreams)
include_directories(${Boost_INCLUDE_DIR})
add_definitions(-DBOOST_NO_AUTO_PTR)

include(CheckEndian)

#include(llvm-extras)
#always import c99 stdint functions into c++
#include(UseStdMacro) # HandleLLVMOptions.cmake (via llvm-extras) already does this for us
#include_directories("${PROJECT_BINARY_DIR}/include") # llvm-extra already does this for us

###### Build Hacks ######

add_definitions(-D__STDC_LIMIT_MACROS)
add_definitions(-D__STDC_CONSTANT_MACROS)

###### Test Inputs ######
if (GRAPH_LOCATION)
  if (EXISTS ${GRAPH_LOCATION}/graph_readme.txt)
    set(BASEINPUT ${GRAPH_LOCATION})
    MESSAGE(STATUS "Using specified graph location ${GRAPH_LOCATION}")
  else()
    MESSAGE(FATAL_ERROR "Specified graph location does not appear to have input: ${GRAPH_LOCATION}")
  endif()
elseif(EXISTS /net/ohm/export/cdgc/inputs/reference/graph_readme.txt)
  set(BASEINPUT /net/ohm/export/cdgc/inputs/reference)
  MESSAGE(STATUS "Using CDGC shared graph folder for inputs")
elseif(EXISTS /net/ohm/export/iss/inputs/reference/graph_readme.txt)
  set(BASEINPUT /net/ohm/export/iss/inputs/reference)
  MESSAGE(STATUS "Using ISS shared graph folder for inputs")
else()
  add_subdirectory(inputs)
  set(BASEINPUT ${CMAKE_BINARY_DIR}/inputs/reference )
  MESSAGE(STATUS "use make input to download inputs in the build directory")
endif()
set(GRAPH_INPUTS ${BASEINPUT}/graph_readme.txt)

###### Global Functions ######

function(add_internal_library name)
  add_library(${name} OBJECT ${ARGN})
endfunction()

include(ProcessorCount)
function(add_test_scale type app)
  processorCount(N)
  if (N EQUAL 0)
    set(N 1)
  endif()
  foreach(thr RANGE 1 ${N})
    add_test(test-${type}-${thr}-${app} ${app} ${ARGN} -t ${thr})
    set_tests_properties(test-${type}-${thr}-${app} PROPERTIES REQUIRED_FILES ${GRAPH_INPUTS})
  endforeach(thr)
endfunction(add_test_scale)

function(compileApp name)
  if(ARGN)
    set(Sources ${ARGN})
  else()
    file(GLOB Sources *.cpp)
  endif()
  add_executable(${name} ${Sources})
endfunction(compileApp)

function(linkApp name extlibs)
  # if(${APP_EXP_OPT})
    # target_link_libraries(${name} galois_exp)
  # endif()
  if (NOT extlibs STREQUAL "")
    target_link_libraries(${name} ${extlibs})
  endif()
  target_link_libraries(${name} lonestar)
  target_link_libraries(${name} galois_shmem)
  target_link_libraries(${name} gllvm)
  target_link_libraries(${name} rt)
  target_link_libraries(${name} ${CMAKE_THREAD_LIBS_INIT})

  if(VTune_FOUND)
    target_link_libraries(${name} ${VTune_LIBRARIES})
  endif()

  if(PAPI_FOUND)
    target_link_libraries(${name} ${PAPI_LIBRARIES})
  endif()

endfunction(linkApp)

# TODO: sepearte out shared and dist libraries to link
# one way to go about it is to create two functions
function(app name)
  set(options DISTSAFE EXP_OPT)
  set(oneValueArgs)
  set(multiValueArgs REQUIRES EXTLIBS)
  cmake_parse_arguments(APP "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  foreach(required ${APP_REQUIRES})
    if(${${required}} MATCHES "TRUE")
    else()
      message(STATUS "NOT compiling ${name} (missing: ${required})")
      return()
    endif()
  endforeach()

  compileApp(${name} ${APP_UNPARSED_ARGUMENTS})
  linkApp(${name} "${APP_EXTLIBS}")

  if(INSTALL_APPS)
    install(TARGETS ${name} DESTINATION bin)
  endif()
endfunction(app)

function(makeTest)
  set(options DISTSAFE ADD_TARGET TARGET EXP_OPT)
  set(oneValueArgs)
  set(multiValueArgs REQUIRES COMMAND_PREFIX)
  cmake_parse_arguments(APP "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  list(GET APP_UNPARSED_ARGUMENTS 0 name)
  list(REMOVE_AT APP_UNPARSED_ARGUMENTS 0)

  foreach(required ${APP_REQUIRES})
    if(${${required}} MATCHES "TRUE")
    else()
      message(STATUS "NOT compiling ${name} (missing: ${required})")
      return()
    endif()
  endforeach()

  set(commandline)
  foreach(f ${APP_COMMAND_PREFIX})
    list(APPEND commandline ${f})
  endforeach()

  if(${APP_ADD_TARGET})
    add_executable(test-${name} ${name}.cpp)
    linkApp(test-${name} "")

    list(APPEND commandline "$<TARGET_FILE:test-${name}>")
  elseif(${APP_TARGET}) 
    get_filename_component(base ${name} NAME)
    if(TARGET ${base})
      list(APPEND commandline "$<TARGET_FILE:${base}>")
    else()
      message(STATUS "NOT adding test ${name} (missing target: ${base})")
      return()
    endif()
  else()
    list(APPEND commandline ${name})
  endif()
  
  foreach(f ${APP_UNPARSED_ARGUMENTS})
    list(APPEND commandline ${f})
  endforeach()

  add_test(NAME ${name} COMMAND ${commandline})

  # Allow parallel tests
  set_tests_properties(${name} PROPERTIES ENVIRONMENT
    GALOIS_DO_NOT_BIND_THREADS=1)
endfunction()

###### Source finding ######
#include_directories(include)

add_subdirectory(libllvm)
add_subdirectory(libgalois)
if(ENABLE_DIST_GALOIS)
  add_subdirectory(libnet)
  add_subdirectory(libdist)
  if(ENABLE_HETERO_GALOIS)
    add_subdirectory(libgpu)
  endif(ENABLE_HETERO_GALOIS)
endif(ENABLE_DIST_GALOIS)
add_subdirectory(tools)
add_subdirectory(scripts)

if(NOT SKIP_COMPILE_APPS)
  add_subdirectory(lonestar)
  if(ENABLE_DIST_GALOIS)
    add_subdirectory(dist_apps)
  endif(ENABLE_DIST_GALOIS)
  add_subdirectory(test)
#  add_subdirectory(toolkit_ml)

  if(USE_DIST_EXP)
    if(ENABLE_DIST_GALOIS)
      add_subdirectory(dist_apps/experimental)
      add_subdirectory(dist_apps/experimental/resilience)
      add_subdirectory(dist_apps/experimental/on_demand)
    endif(ENABLE_DIST_GALOIS)
  endif(USE_DIST_EXP)
  if(USE_EXP)
    add_subdirectory(lonestar/experimental)
  endif(USE_EXP)
endif()

###### Documentation ######
find_package(Doxygen)
if(DOXYGEN_FOUND)
  #TODO: double check the list of directories here
  set(DOXYFILE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/docs\" \"${CMAKE_CURRENT_SOURCE_DIR}/libgalois\" \"${CMAKE_CURRENT_SOURCE_DIR}/libdist")
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in
     ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile.in @ONLY)
  add_custom_target(doc ${DOXYGEN_EXECUTABLE}
     ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile.in WORKING_DIRECTORY
     ${CMAKE_CURRENT_BINARY_DIR})
endif()

###### Distribution ######
include(InstallRequiredSystemLibraries)
set(CPACK_GENERATOR "TGZ")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/COPYRIGHT")
set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
set(CPACK_PACKAGE_VERSION_MAJOR ${GALOIS_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${GALOIS_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${GALOIS_VERSION_PATCH})
include(CPack)

###### Installation ######
#export(TARGETS galois APPEND FILE "${PROJECT_BINARY_DIR}/GaloisTargets.cmake")
#export(PACKAGE Galois)

# Galois compiler definitions
set(galois_defs)
get_directory_property(defs DIRECTORY "${PROJECT_SOURCE_DIR}" COMPILE_DEFINITIONS)
foreach(d ${defs})
  set(galois_defs "-D${d} ${galois_defs}")
endforeach()
get_directory_property(defs DIRECTORY "${PROJECT_SOURCE_DIR}" COMPILE_DEFINITIONS_${CMAKE_BUILD_TYPE})
foreach(d ${defs})
  set(galois_defs "-D${d} ${galois_defs}")
endforeach()
string(TOUPPER ${CMAKE_BUILD_TYPE} upper_build_type)
set(GALOIS_FLAGS "${galois_defs} ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${upper_build_type}}")
set(GALOIS_CXX_COMPILER "${CMAKE_CXX_COMPILER}")

file(RELATIVE_PATH PATH_FROM_INSTALLED_CMAKE_FILE_TO_PREFIX "${INSTALL_CMAKE_DIR}" "${CMAKE_INSTALL_PREFIX}")

# Generate appropriate CMake files for installation and build trees
configure_file("${PROJECT_SOURCE_DIR}/cmake/Modules/GaloisConfig.cmake.in"
  "${PROJECT_BINARY_DIR}/GaloisConfig.cmake" @ONLY)
configure_file("${PROJECT_SOURCE_DIR}/cmake/Modules/GaloisConfig.cmake.in"
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/GaloisConfig.cmake" @ONLY)
configure_file("${PROJECT_SOURCE_DIR}/cmake/Modules/GaloisConfigVersion.cmake.in"
  "${PROJECT_BINARY_DIR}/GaloisConfigVersion.cmake" @ONLY)
install(FILES
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/GaloisConfig.cmake"
  "${PROJECT_BINARY_DIR}/GaloisConfigVersion.cmake"
  DESTINATION "${INSTALL_CMAKE_DIR}" COMPONENT dev)
install(EXPORT GaloisTargets DESTINATION "${INSTALL_CMAKE_DIR}" COMPONENT dev)
