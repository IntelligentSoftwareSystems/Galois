cmake_minimum_required(VERSION 3.17)

project(Galois)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")

include(GNUInstallDirs)

if(STACK_CAPTURE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -finstrument-functions")
  set(INSTRUMENT_EXCLUDE_FILE "${CMAKE_CURRENT_SOURCE_DIR}/libgalois/include/galois/runtime/StackTracer.h")
  set(INSTRUMENT_EXCLUDE_FILE "${INSTRUMENT_EXCLUDE_FILE},/usr/include/c++/11/sstream")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -finstrument-functions-exclude-file-list=${INSTRUMENT_EXCLUDE_FILE}")
endif(STACK_CAPTURE)

file(STRINGS config/version.txt GALOIS_VERSION)
string(REGEX REPLACE "[ \t\n]" "" GALOIS_VERSION ${GALOIS_VERSION})
string(REGEX REPLACE "([0-9]+)\\.([0-9]+)\\.([0-9]+)" "\\1" GALOIS_VERSION_MAJOR ${GALOIS_VERSION})
string(REGEX REPLACE "([0-9]+)\\.([0-9]+)\\.([0-9]+)" "\\2" GALOIS_VERSION_MINOR ${GALOIS_VERSION})
string(REGEX REPLACE "([0-9]+)\\.([0-9]+)\\.([0-9]+)" "\\3" GALOIS_VERSION_PATCH ${GALOIS_VERSION})
set(GALOIS_COPYRIGHT_YEAR "2018") # Also in COPYRIGHT

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "No build type selected, default to Release")
  # cmake default flags with relwithdebinfo is -O2 -g
  # cmake default flags with release is -O3 -DNDEBUG
  set(CMAKE_BUILD_TYPE "Release")
endif()

###### Options (alternatively pass as options to cmake -DName=Value) ######
###### Distributed-heterogeneous features ######
set(GALOIS_ENABLE_DIST ON CACHE BOOL "Enable distributed features")
set(GALOIS_CUDA_CAPABILITY "" CACHE STRING "Semi-colon list of CUDA compute capability version numbers to enable GPU features") # e.g., "3.7;6.1"
set(GALOIS_COMM_STATS OFF CACHE BOOL "Report more detailed statistics of communication")
set(GALOIS_ENABLE_WMD ON CACHE BOOL "Enable WMD dataset support")
###### General features ######
set(GALOIS_ENABLE_PAPI OFF CACHE BOOL "Use PAPI counters for profiling")
set(GALOIS_ENABLE_VTUNE OFF CACHE BOOL "Use VTune for profiling")
set(GALOIS_STRICT_CONFIG OFF CACHE BOOL "Instead of falling back gracefully, fail")
set(GALOIS_GRAPH_LOCATION "" CACHE PATH "Location of inputs for tests if downloaded/stored separately.")
set(CXX_CLANG_TIDY "" CACHE STRING "Semi-colon list specifying clang-tidy command and arguments")
set(CMAKE_CXX_COMPILER_LAUNCHER "" CACHE STRING "Semi-colon list specifying command to wrap compiler invocations (e.g., ccache)")
set(USE_ARCH native CACHE STRING "Optimize for a specific processor architecture ('none' to disable)")

set(USE_DEEPGALOIS OFF CACHE BOOL "Use gnn apps as well as the DeepGalois library")
set(USE_MKL_BLAS OFF CACHE BOOL "Use MKL for BLAS")
# TODO; this is GNN related; find better way to do than hardcode
#SET(CUDA_HOME /org/centers/cdgc/cuda/cuda-10.2)

# This option is automatically handled by CMake.
# It makes add_library build a shared lib unless STATIC is explicitly specified.
# Putting this here is mostly just a placeholder so people know it's an option.
# Currently this is really only intended to change anything for the libgalois_shmem target.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libraries")
###### Developer features ######
set(GALOIS_PER_ROUND_STATS OFF CACHE BOOL "Report statistics of each round of execution")
set(GALOIS_NUM_TEST_GPUS "0" CACHE STRING "Number of test GPUs to use (on a single machine) for running the tests.")
set(GALOIS_USE_LCI OFF CACHE BOOL "Use LCI network runtime instead of MPI")
set(GALOIS_USE_BARE_MPI OFF CACHE BOOL "Use MPI directly (no dedicated network-runtime thread)")
set(GALOIS_NUM_TEST_THREADS "" CACHE STRING "Maximum number of threads to use when running tests (default: number of physical cores)")
set(GALOIS_ENABLE_INSTRUMENT OFF CACHE BOOL "Enable generating instrument in the runtime")

if(NOT GALOIS_NUM_TEST_THREADS)
  cmake_host_system_information(RESULT GALOIS_NUM_TEST_THREADS QUERY NUMBER_OF_PHYSICAL_CORES)
endif()
if(GALOIS_NUM_TEST_THREADS LESS_EQUAL 0)
  set(GALOIS_NUM_TEST_THREADS 1)
endif()

###### Configure (users don't need to go beyond here) ######

include(CTest)

###### Configure compiler ######

if(PROJECT_IS_TOP_LEVEL)
  if(CMAKE_CXX_FLAGS)
    message(STATUS "Provided CXX Flags: " ${CMAKE_CXX_FLAGS})
  endif()

  set(CMAKE_OPTIMIZE_DEPENDENCIES true)

  # Setup CCache
  find_program(CCACHE_EXECUTABLE ccache)
  if(CCACHE_EXECUTABLE)
    message(STATUS "CCache found at: " ${CCACHE_EXECUTABLE})
    set(ccacheEnv
      CCACHE_SLOPPINESS=pch_defines,time_macros
    )
    # NOTE: Ccache 4.2+ required for reliable CUDA support
    foreach(lang IN ITEMS C CXX OBJC OBJCXX CUDA)
      set(CMAKE_${lang}_COMPILER_LAUNCHER
        ${CMAKE_COMMAND} -E env ${ccacheEnv} ${CCACHE_EXECUTABLE}
      )
    endforeach()
  endif()
endif()

# generate compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) #...without compiler extensions like gnu++11
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Always include debug info
add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-g>")

# GCC
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7)
    message(FATAL_ERROR "gcc must be version 7 or higher. Found ${CMAKE_CXX_COMPILER_VERSION}.")
  endif()

  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra>")

  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 11)
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Werror>")
  endif()
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7)
    message(FATAL_ERROR "clang must be version 7 or higher. Found ${CMAKE_CXX_COMPILER_VERSION}.")
  endif()

  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra>")

  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 11)
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Werror>")
  endif()
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra>")

  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 12)
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Werror>")
  endif()
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 19.0.1)
    message(FATAL_ERROR "icpc must be 19.0.1 or higher. Found ${CMAKE_CXX_COMPILER_VERSION}.")
  endif()

  # Avoid warnings when using noinline for methods defined inside class defintion.
  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-wd2196>")
endif()

# Enable architecture-specific optimizations
include(CheckArchFlags)
if(ARCH_FLAGS_FOUND)
  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:${ARCH_CXX_FLAGS}>")
  add_compile_options("$<$<COMPILE_LANGUAGE:C>:${ARCH_C_FLAGS}>")
  add_link_options(${ARCH_LINK_FLAGS})
endif()

if(CXX_CLANG_TIDY)
  set(CMAKE_CXX_CLANG_TIDY ${CXX_CLANG_TIDY} "-header-filter=.*${PROJECT_SOURCE_DIR}.*")
  # Ignore warning flags intended for the CXX program. This only works because
  # the two compilers we care about, clang and gcc, both understand
  # -Wno-unknown-warning-option.
  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wno-unknown-warning-option>")
endif()

###### Configure features ######

################################################################################
# For GNN matrix multiplies
# TODO (loc) prefix with GALOIS, move elsewhere more fitting in this file
################################################################################
if(USE_MKL_BLAS)
  find_package(MKL CONFIG REQUIRED PATH $ENV{MKL_ROOT})
  if (MKL_FOUND)
  else()
    message(WARNING "MKL not found")
  endif()
endif()

#SET(OPENBLAS_ROOT /org/centers/cdgc/openblas/gcc8.1)
if(USE_OPENBLAS)
  find_package(OpenBLAS)
  message(STATUS "OpenBLAS: ${OPENBLAS_INCLUDE_DIRS}")
  if (OPENBLAS_FOUND)
    include_directories(${OPENBLAS_INCLUDE_DIRS})
  else()
    message(WARNING "OpenBLAS not found")
  endif()
endif()

################################################################################

if(GALOIS_ENABLE_VTUNE)
  set(VTune_ROOT /opt/intel/vtune_amplifier)
  find_package(VTune REQUIRED)
  include_directories(${VTune_INCLUDE_DIRS})
  add_definitions(-DGALOIS_ENABLE_VTUNE)
endif()


if(GALOIS_ENABLE_PAPI)
  find_package(PAPI REQUIRED)
  include_directories(${PAPI_INCLUDE_DIRS})
  add_definitions(-DGALOIS_ENABLE_PAPI)
endif()

if (GALOIS_ENABLE_INSTRUMENT)
  add_definitions(-DGALOIS_INSTRUMENT)
endif()

find_package(Threads REQUIRED)

include(CheckMmap)

include(CheckHugePages)
if(NOT HAVE_HUGEPAGES AND GALOIS_STRICT_CONFIG)
  message(FATAL_ERROR "Need huge pages")
endif()

find_package(Boost 1.58.0 REQUIRED COMPONENTS serialization iostreams)

find_package(LLVM REQUIRED CONFIG)
if("${LLVM_PACKAGE_VERSION}" VERSION_LESS "7")
  message(FATAL_ERROR "LLVM 7 or greater is required.")
endif()
if(NOT DEFINED LLVM_ENABLE_RTTI)
  message(FATAL_ERROR "Could not determine if LLVM has RTTI enabled.")
endif()
if(NOT ${LLVM_ENABLE_RTTI})
  message(FATAL_ERROR "Galois requires a build of LLVM that includes RTTI. Most package managers do this already, but if you built LLVM from source you need to configure it with `-DLLVM_ENABLE_RTTI=ON`")
endif()
target_include_directories(LLVMSupport INTERFACE ${LLVM_INCLUDE_DIRS})

include(HandleSanitizer)

include(CheckEndian)

###### Test Inputs ######

if(GALOIS_GRAPH_LOCATION)
  set(BASEINPUT "${GALOIS_GRAPH_LOCATION}")
  set(BASEOUTPUT "${GALOIS_GRAPH_LOCATION}")
  message(STATUS "Using graph input and output location ${GALOIS_GRAPH_LOCATION}")
elseif(EXISTS /net/ohm/export/iss)
  set(BASEINPUT /net/ohm/export/iss/inputs)
  MESSAGE(STATUS "Using graph input location /net/ohm/export/iss/inputs")
  set(BASEOUTPUT /net/ohm/export/iss/dist-outputs)
  MESSAGE(STATUS "Using graph output location /net/ohm/export/iss/dist-outputs")
else()
  set(BASEINPUT "${PROJECT_BINARY_DIR}/inputs")
  set(BASEOUTPUT "${PROJECT_BINARY_DIR}/inputs")
  message(STATUS "Use 'make input' to download inputs and outputs in the build directory")
endif()

###### Source finding ######

add_custom_target(lib)
add_custom_target(apps)

# Core libraries (lib)
add_subdirectory(libsupport)
add_subdirectory(libgalois)
add_subdirectory(libpygalois)
if (GALOIS_ENABLE_DIST)
  find_package(MPI REQUIRED)
  add_subdirectory(libdist)
  add_subdirectory(libcusp)
  add_subdirectory(libgluon)
  if (GALOIS_ENABLE_WMD)
    find_package(MPI REQUIRED)
    add_subdirectory(libwmd)
  endif()
endif()

# TODO(loc) prefix with GALOIS
if(USE_DEEPGALOIS)
  add_subdirectory(libdeepgalois)
endif(USE_DEEPGALOIS)

string(COMPARE NOTEQUAL "${GALOIS_CUDA_CAPABILITY}" "" GALOIS_ENABLE_GPU)

if (GALOIS_ENABLE_GPU)
  enable_language(CUDA)
  foreach(GENCODE ${GALOIS_CUDA_CAPABILITY})
    string(REPLACE "." "" GENCODE ${GENCODE})
    add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_${GENCODE},code=sm_${GENCODE}>")
  endforeach()

  # This is necessary to allow building for CUDA 11.x (where CUB is bundled) and earlier versions (where CUB is not included)
  add_definitions(-DTHRUST_IGNORE_CUB_VERSION_CHECK)

  add_subdirectory(libgpu)

  if (USE_DEEPGALOIS)
    SET(CUDA_SEPARABLE_COMPILATION ON)
    find_package(CUDA REQUIRED)
    set(CUDA_PROPAGATE_HOST_FLAGS off)
    set(CUDA_HOST_COMPILER g++)

    string(REPLACE "." "" GENCODES ${GALOIS_CUDA_CAPABILITY})
    string(REPLACE "," ";" GENCODES ${GENCODES})
    foreach(GENCODE ${GENCODES})
      set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; --expt-extended-lambda -gencode arch=compute_${GENCODE},code=sm_${GENCODE})
    endforeach()

    cuda_include_directories("${CMAKE_SOURCE_DIR}/libgpu/include")

    # MGPU v1.1
    set(MGPU_ROOT "${CMAKE_SOURCE_DIR}/moderngpu") # only required headers
    cuda_include_directories("${MGPU_ROOT}/src")

    # CUB v1.6.4
    set(CUB_ROOT "${CMAKE_SOURCE_DIR}/cub") # only required headers
    cuda_include_directories("${CUB_ROOT}")

    #find_package(OpenCL REQUIRED)
  endif()
endif()

if (GALOIS_ENABLE_DIST AND USE_MKL_BLAS)
  # here because I need the GPU declarations above
  add_subdirectory(libgnn)
endif()

add_subdirectory(libpangolin)

# Applications (apps)
add_subdirectory(lonestar)

add_subdirectory(scripts)
add_subdirectory(inputs)
add_subdirectory(tools)

if(USE_EXP)
  add_subdirectory(lonestar/experimental)
endif(USE_EXP)

###### Documentation ######

find_package(Doxygen)
if(DOXYGEN_FOUND)
  #TODO: double check the list of directories here
  set(DOXYFILE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/docs\" \"${CMAKE_CURRENT_SOURCE_DIR}/libcusp\" \"${CMAKE_CURRENT_SOURCE_DIR}/libdist\" \"${CMAKE_CURRENT_SOURCE_DIR}/libgalois/src\" \"${CMAKE_CURRENT_SOURCE_DIR}/libgalois/include\" \"${CMAKE_CURRENT_SOURCE_DIR}/libgluon")
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in
     ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile.in @ONLY)
  add_custom_target(doc ${DOXYGEN_EXECUTABLE}
     ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile.in WORKING_DIRECTORY
     ${CMAKE_CURRENT_BINARY_DIR})
endif()

###### Installation ######

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/GaloisConfigVersion.cmake
  VERSION ${GALOIS_VERSION}
  COMPATIBILITY SameMajorVersion
)
configure_package_config_file(
  cmake/GaloisConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/GaloisConfig.cmake
  INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/Galois"
  PATH_VARS CMAKE_INSTALL_INCLUDEDIR CMAKE_INSTALL_LIBDIR CMAKE_INSTALL_BINDIR
)
install(
  FILES "${CMAKE_CURRENT_BINARY_DIR}/GaloisConfigVersion.cmake" "${CMAKE_CURRENT_BINARY_DIR}/GaloisConfig.cmake"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/Galois"
  COMPONENT dev
)
install(
  EXPORT GaloisTargets
  NAMESPACE Galois::
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/Galois"
  COMPONENT dev
)

###### Distribution ######

set(CPACK_GENERATOR "TGZ")
set(CPACK_ARCHIVE_COMPONENT_INSTALL ON)
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/COPYRIGHT")
set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
set(CPACK_PACKAGE_VERSION_MAJOR ${GALOIS_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${GALOIS_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${GALOIS_VERSION_PATCH})
include(CPack)

if(STACK_CAPTURE)
  message("Writing CMAKE_CXX_FLAGS")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSTACK_TRACE -finstrument-functions")
  set(INSTRUMENT_EXCLUDE_FILE "${CMAKE_CURRENT_SOURCE_DIR}/libgalois/include/galois/runtime/StackTracer.h")
  set(INSTRUMENT_EXCLUDE_FILE "${INSTRUMENT_EXCLUDE_FILE},/usr/include/c++/11/sstream")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -finstrument-functions-exclude-file-list=${INSTRUMENT_EXCLUDE_FILE}" CACHE STRING "CMAKE Flags" FORCE)
endif(STACK_CAPTURE)
