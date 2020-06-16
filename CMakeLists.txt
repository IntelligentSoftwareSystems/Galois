cmake_minimum_required(VERSION 3.13)

project(Galois)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")

include(GNUInstallDirs)

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
set(GALOIS_ENABLE_DIST OFF CACHE BOOL "Enable distributed features")
set(GALOIS_CUDA_CAPABILITY "" CACHE STRING "Semi-colon list of CUDA compute capability version numbers to enable GPU features") # e.g., "3.7;6.1"
set(GALOIS_COMM_STATS OFF CACHE BOOL "Report more detailed statistics of communication")
###### General features ######
set(GALOIS_ENABLE_PAPI OFF CACHE BOOL "Use PAPI counters for profiling")
set(GALOIS_ENABLE_VTUNE OFF CACHE BOOL "Use VTune for profiling")
set(GALOIS_STRICT_CONFIG OFF CACHE BOOL "Instead of falling back gracefully, fail")
set(GALOIS_GRAPH_LOCATION "" CACHE PATH "Location of inputs for tests if downloaded/stored separately.")
set(CXX_CLANG_TIDY "" CACHE STRING "Semi-colon list specifying clang-tidy command and arguments")
set(CMAKE_CXX_COMPILER_LAUNCHER "" CACHE STRING "Semi-colon list specifying command to wrap compiler invocations (e.g., ccache)")
set(USE_ARCH native CACHE STRING "Optimize for a specific processor architecture ('none' to disable)")
set(GALOIS_USE_SANITIZER "" CACHE STRING "Semi-colon list of sanitizers to use (Memory, MemoryWithOrigins, Address, Undefined, Thread)")
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

if(NOT GALOIS_NUM_TEST_THREADS)
  cmake_host_system_information(RESULT GALOIS_NUM_TEST_THREADS QUERY NUMBER_OF_PHYSICAL_CORES)
endif()
if(GALOIS_NUM_TEST_THREADS LESS_EQUAL 0)
  set(GALOIS_NUM_TEST_THREADS 1)
endif()

###### Configure (users don't need to go beyond here) ######

include(CTest)

###### Configure compiler ######

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
add_subdirectory(libgalois)
if (GALOIS_ENABLE_DIST)
  find_package(MPI REQUIRED)
  add_subdirectory(libdist)
  add_subdirectory(libcusp)
  add_subdirectory(libgluon)
endif()
string(COMPARE NOTEQUAL "${GALOIS_CUDA_CAPABILITY}" "" GALOIS_ENABLE_GPU)
if (GALOIS_ENABLE_GPU)
  enable_language(CUDA)
  foreach(GENCODE ${GALOIS_CUDA_CAPABILITY})
    string(REPLACE "." "" GENCODE ${GENCODE})
    add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_${GENCODE},code=sm_${GENCODE}>")
  endforeach()

  add_subdirectory(libgpu)
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
