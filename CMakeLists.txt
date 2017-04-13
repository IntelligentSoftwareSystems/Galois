cmake_minimum_required(VERSION 3.2.0)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")
project(Galois)
set(GALOIS_VERSION_MAJOR "2")
set(GALOIS_VERSION_MINOR "3")
set(GALOIS_VERSION_PATCH "0")
set(GALOIS_VERSION ${GALOIS_VERSION_MAJOR}.${GALOIS_VERSION_MINOR}.${GALOIS_VERSION_PATCH})
set(GALOIS_COPYRIGHT_YEAR "2015") # Also in COPYRIGHT

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
set(USE_GPROF OFF CACHE BOOL "Enable GCC profiling")
set(USE_VTUNE OFF CACHE BOOL "Use VTune for profiling")
set(USE_PAPI OFF CACHE BOOL "Use PAPI counters for profiling")
set(USE_HPCTOOLKIT OFF CACHE BOOL "Use HPCToolKit for profiling")
set(USE_STRICT_CONFIG OFF CACHE BOOL "Instead of falling back gracefully, fail")
set(USE_LONGJMP ON CACHE BOOL "Use longjmp instead of exceptions to signal aborts")
set(USE_SANATIZER OFF CACHE BOOL "Use address and memory sanatizer")
set(INSTALL_APPS OFF CACHE BOOL "Install apps as well as library")
set(SKIP_COMPILE_APPS OFF CACHE BOOL "Skip compilation of applications using Galois library")
set(GRAPH_LOCATION "" CACHE PATH "Location of inputs for tests if downloaded/stored separately.")

set(INSTALL_LIB_DIR lib CACHE PATH "Installation directory for libraries")
set(INSTALL_BIN_DIR bin CACHE PATH "Installation directory for executables")
set(INSTALL_INCLUDE_DIR include CACHE PATH "Installation directory for header files")
set(INSTALL_CMAKE_DIR lib/cmake/Galois CACHE PATH "Installation directory for CMake files")
# Make relative paths absolute
foreach(p LIB BIN INCLUDE CMAKE)
  set(var INSTALL_${p}_DIR)
  if(NOT IS_ABSOLUTE "${${var}}")
    set(${var} "${CMAKE_INSTALL_PREFIX}/${${var}}")
  endif()
endforeach()

# Enable iss specific options; should be OFF in the general release; all guarded by USE_EXP
set(USE_EXP OFF CACHE BOOL "Use experimental features")
set(USE_LWCI OFF CACHE BOOL "Use LWCI communication runtime")
set(USE_HTM OFF CACHE BOOL "Use HTM")
set(EXP_DOALL "PTHREAD" CACHE STRING "Which type of implementation of parallel_doall")
set(USE_PROF OFF CACHE BOOL "Use profiling specific features")
set(USE_EMBEDDED_REVISION ON CACHE BOOL "Embed revision numbers")
set(USE_HETERO ON CACHE BOOL "Use Heterogeneous features")
set(USE_GGC OFF CACHE BOOL "Use IrGL Compiler (CUDA)")

###### Configure (users don't need to go beyond here) ######

enable_testing()

###### Configure compiler ######

#Always include debug info and aggressive warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -Wall")

if(NOT "$ENV{GCC_BIN}" STREQUAL "")
  # Preserve captured environment variable during reconfiguration
  set(CXX_TOOLCHAIN "$ENV{GCC_BIN}" CACHE STRING "GCC Toolchain")
endif()

# ICC
if(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
  execute_process(COMMAND ${CMAKE_CXX_COMPILER} -dumpversion
                  OUTPUT_VARIABLE ICC_VERSION)
  if(ICC_VERSION VERSION_LESS 13.0)
    message(FATAL_ERROR "ICC must be 13.0 or higher; found: ${ICC_VERSION} instead")
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
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -U__STRICT_ANSI__ -Werror")
endif()

# check for incompatible GCC
if(CMAKE_COMPILER_IS_GNUCC)
  execute_process(COMMAND ${CMAKE_CXX_COMPILER} -dumpversion
                  OUTPUT_VARIABLE GCC_VERSION)
  if(GCC_VERSION VERSION_LESS 4.7)
    message(FATAL_ERROR "GCC must be 4.7 or higher")
  endif()
  #new debugging optimization level
  if(CMAKE_BUILD_TYPE MATCHES "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Og")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Og")
  endif()
endif()

# Enable architecture-specific optimizations
if(NOT USE_HETERO)
  find_package(ArchFlags)
  if(ARCH_FLAGS_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ARCH_CXX_FLAGS}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${ARCH_C_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${ARCH_EXE_FLAGS}")
  endif()
endif()

# More warnings
if(CMAKE_BUILD_TYPE MATCHES "Debug")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")
  if(CMAKE_COMPILER_IS_GNUCC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Og")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Og")
  endif()
endif()

# GNU profiling
if(USE_GPROF)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pg")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
endif(USE_GPROF)

if(USE_SANATIZER)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer -fno-optimize-sibling-calls")
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

###### Configure features ######

add_definitions(-DGALOIS_VERSION_MAJOR=${GALOIS_VERSION_MAJOR})
add_definitions(-DGALOIS_VERSION_MINOR=${GALOIS_VERSION_MINOR})
add_definitions(-DGALOIS_VERSION_PATCH=${GALOIS_VERSION_PATCH})
add_definitions(-DGALOIS_VERSION=${GALOIS_VERSION_MAJOR}.${GALOIS_VERSION_MINOR}.${GALOIS_VERSION_PATCH})
add_definitions(-DGALOIS_COPYRIGHT_YEAR=${GALOIS_COPYRIGHT_YEAR})

# Experimental features
if(USE_EXP)
  set(USE_VTUNE ON)
  add_definitions(-DGALOIS_USE_EXP)
  include_directories("libexp/include")

  if (USE_LWCI)
    add_definitions(-DGALOIS_USE_LWCI)
    #include_directories("$ENV{LWCI_ROOT}/include")
    set(LWCI_LIBRARY "$ENV{LWCI_ROOT}/lib/liblwci.a")
  endif()

  find_package(MPI REQUIRED) # for distributed
  set(CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS})
  set(CMAKE_CXX_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS})
  include_directories(${MPI_INCLUDE_PATH})

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
      set(GALOIS_USE_LONGJMP on)
    else()
      message(FATAL_ERROR "Hardware transactional memory not supported")
    endif()
  endif()
  if(USE_HETERO)
    find_package(CUDA REQUIRED)
    #find_package(OpenCL REQUIRED)
    if(CUDA_FOUND)
      set(CUDA_PROPAGATE_HOST_FLAGS off)
      set(CUDA_SEPARABLE_COMPILATION on)
      set(CUDA_HOST_COMPILER g++)

      if(USE_GGC)
        message("INSIDE GCC")
        if(NOT "$ENV{GGC_PATH}" STREQUAL "")
          set(GGC_PATH $ENV{GGC_PATH})
          message(STATUS "GGC_PATH set to ${GGC_PATH}")
        else()
          set(GGC_PATH "/net/velocity/workspace/SourceCode/ggc")
        endif()
        if (EXISTS ${GGC_PATH})
        else()
          message(FATAL_ERROR "GGC_PATH=${GGC_PATH} does not exist; set environment variable GGC_PATH to override")
        endif()

        set(GGC_INCLUDE_PATH "${GGC_PATH}/rt/include")
        set(MGPU_PATH "${GGC_INCLUDE_PATH}/mgpu/include")

        cuda_include_directories( ${GGC_INCLUDE_PATH} ${MGPU_PATH})
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61)
        link_directories(${GGC_PATH}/rt/lib)
      endif()
    else()
      message(FATAL_ERROR "Need CUDA for heterogenous code")
    endif()
  endif()
endif()

if(USE_VTUNE)
  message(STATUS "${VTune_ROOT}")
  set(VTune_ROOT /opt/intel/vtune_amplifier_xe)
  find_package(VTune)
  message(STATUS "${VTune_INCLUDE_DIRS}")

  if (VTune_FOUND)
    include_directories(${VTune_INCLUDE_DIRS})
    add_definitions(-DGALOIS_USE_VTUNE)
  else()
    message(WARNING "VTune not found")
  endif()
endif()

# PThreads
find_package(Threads REQUIRED)

#llvm ADT and command line parser
#find_package(LLVM)

include(CheckCilk)
include(CheckMmap)

# HugePages
include(CheckHugePages)
if(NOT HAVE_HUGEPAGES AND USE_STRICT_CONFIG)
  message(FATAL_ERROR "Need huge pages")
endif()

# Longjmp
if(USE_LONGJMP)
  set(GALOIS_USE_LONGJMP on)
endif()

# Boost
set(Boost_ADDITIONAL_VERSIONS "1.40" "1.40.0" "1.47" "1.47.0" "1.49" "1.49.0" "1.51.0")
set(Boost_USE_MULTITHREADED OFF)
if(NOT "$ENV{BOOST_DIR}" STREQUAL "")
  set(BOOST_ROOT $ENV{BOOST_DIR})
endif()
find_package(Boost 1.38.0 REQUIRED)
include_directories(${Boost_INCLUDE_DIR})

include(CheckEndian)

#include(llvm-extras)
#always import c99 stdint functions into c++
#include(UseStdMacro) # HandleLLVMOptions.cmake (via llvm-extras) already does this for us
#include_directories("${PROJECT_BINARY_DIR}/include") # llvm-extra already does this for us

###### Build Hacks ######

add_definitions(-D__STDC_LIMIT_MACROS)

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
  if(${APP_EXP_OPT})
    target_link_libraries(${name} galois_exp)
  endif()
  target_link_libraries(${name} ${APP_EXTLIBS})
  target_link_libraries(${name} lonestar)
  target_link_libraries(${name} galois_net)
  target_link_libraries(${name} galois_graphs)
  target_link_libraries(${name} galois_runtime)
  target_link_libraries(${name} galois_substrate)
  target_link_libraries(${name} gllvm)
  target_link_libraries(${name} rt)
  target_link_libraries(${name} ${CMAKE_THREAD_LIBS_INIT})

  if(VTune_FOUND)
    target_link_libraries(${name} ${VTune_LIBRARIES})
  endif()

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
    if (${APP_EXP_OPT})
      target_link_libraries(test-${name} galois_exp)
    endif()
    target_link_libraries(test-${name} lonestar)
    target_link_libraries(test-${name} galois_net)
    target_link_libraries(test-${name} galois_graphs)
    target_link_libraries(test-${name} galois_runtime)
    target_link_libraries(test-${name} galois_substrate)
    target_link_libraries(test-${name} gllvm)
#    target_link_libraries(test-${name} galois_substrate)
#    target_link_libraries(test-${name} galois)
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
add_subdirectory(libsubstrate)
add_subdirectory(libruntime)
add_subdirectory(libgraphs)
add_subdirectory(libnet)
add_subdirectory(libexp)

add_subdirectory(tools)
add_subdirectory(scripts)
if(NOT SKIP_COMPILE_APPS)
  add_subdirectory(lonestar)
  add_subdirectory(apps)
  add_subdirectory(examples)
  add_subdirectory(test)
#  add_subdirectory(toolkit_ml)
  add_subdirectory(exp)
endif()

###### Documentation ######
find_package(Doxygen)
if(DOXYGEN_FOUND)
set(DOXYFILE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/docs\" \"${CMAKE_CURRENT_SOURCE_DIR}/libgraphs\" \"${CMAKE_CURRENT_SOURCE_DIR}/libllvm\" \"${CMAKE_CURRENT_SOURCE_DIR}/libruntime\" \"${CMAKE_CURRENT_SOURCE_DIR}/libsubstrate")
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
set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README")
set(CPACK_PACKAGE_VERSION_MAJOR ${GALOIS_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${GALOIS_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${GALOIS_VERSION_PATCH})
include(CPack)

###### Installation ######
#export(TARGETS galois APPEND FILE "${PROJECT_BINARY_DIR}/GaloisTargets.cmake")
#export(PACKAGE Galois)

# Galois include files
file(RELATIVE_PATH rel_include_dir "${INSTALL_CMAKE_DIR}" "${INSTALL_INCLUDE_DIR}")
set(GALOIS_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include" "${PROJECT_BINARY_DIR}/include")
#if(USE_EXP)
#  set(GALOIS_INCLUDE_DIR "${GALOIS_INCLUDE_DIR}" "${PROJECT_SOURCE_DIR}/libexp/include" "${PROJECT_SOURCE_DIR}/libgaloiscl/include")
#endif()


set(GALOIS_INCLUDE_DIRS) # Galois include dependencies
set(GALOIS_INCLUDE_DIR)  # Galois include provides
#get_target_property(defs galois INCLUDE_DIRECTORIES)
foreach(d ${defs})
  string(FIND ${d} ${PROJECT_BINARY_DIR} pos1)
  string(FIND ${d} ${PROJECT_SOURCE_DIR} pos2)
  if(NOT ${pos1} EQUAL 0 AND NOT ${pos2} EQUAL 0)
    set(GALOIS_INCLUDE_DIRS ${GALOIS_INCLUDE_DIRS} ${d})
  else()
    set(GALOIS_INCLUDE_DIR ${GALOIS_INCLUDE_DIR} ${d})
  endif()
endforeach()
install(DIRECTORY ${GALOIS_INCLUDE_DIR} DESTINATION "${INSTALL_INCLUDE_DIR}/.." COMPONENT dev
  FILES_MATCHING PATTERN "*.h"
  PATTERN ".svn" EXCLUDE)

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

# Generate appropriate CMake files for installation and build trees
configure_file("${PROJECT_SOURCE_DIR}/cmake/Modules/GaloisConfig.cmake.in"
  "${PROJECT_BINARY_DIR}/GaloisConfig.cmake" @ONLY)
set(GALOIS_INCLUDE_DIR "\${GALOIS_CMAKE_DIR}/${rel_include_dir}")
configure_file("${PROJECT_SOURCE_DIR}/cmake/Modules/GaloisConfig.cmake.in"
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/GaloisConfig.cmake" @ONLY)
configure_file("${PROJECT_SOURCE_DIR}/cmake/Modules/GaloisConfigVersion.cmake.in"
  "${PROJECT_BINARY_DIR}/GaloisConfigVersion.cmake" @ONLY)
install(FILES
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/GaloisConfig.cmake"
  "${PROJECT_BINARY_DIR}/GaloisConfigVersion.cmake"
  DESTINATION "${INSTALL_CMAKE_DIR}" COMPONENT dev)
install(EXPORT GaloisTargets DESTINATION "${INSTALL_CMAKE_DIR}" COMPONENT dev)
