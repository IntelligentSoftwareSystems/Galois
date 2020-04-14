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
set(ENABLE_DIST_GALOIS OFF CACHE BOOL "Enable distributed features")
set(ENABLE_HETERO_GALOIS OFF CACHE BOOL "Enable heterogeneous features")
set(USE_LCI OFF CACHE BOOL "Use LCI network runtime instead of MPI")
set(USE_BARE_MPI OFF CACHE BOOL "Use MPI directly (no dedicated network-runtime thread)")
set(CUDA_CAPABILITY "3.7,6.1" CACHE STRING "Comma-separated CUDA capability version numbers")
set(REPORT_COMM_STATS OFF CACHE BOOL "Report more detailed statistics of communication")
set(REPORT_PER_ROUND_STATS OFF CACHE BOOL "Report statistics of each round of execution")
set(NUM_TEST_GPUS "0" CACHE STRING "Number of test GPUs to use (on a single machine) for running the tests.")
###### General features ######
set(USE_GPROF OFF CACHE BOOL "Enable GCC profiling")
set(USE_VTUNE OFF CACHE BOOL "Use VTune for profiling")
set(USE_PAPI OFF CACHE BOOL "Use PAPI counters for profiling")
set(USE_HPCTK OFF CACHE BOOL "Use HPCToolKit for profiling")
set(USE_STRICT_CONFIG OFF CACHE BOOL "Instead of falling back gracefully, fail")
set(USE_LONGJMP_ABORT ON CACHE BOOL "Use longjmp instead of exceptions to signal aborts")
set(USE_SANITIZER OFF CACHE BOOL "Use address and memory sanatizer")
set(GRAPH_LOCATION "" CACHE PATH "Location of inputs for tests if downloaded/stored separately.")
set(CXX_CLANG_TIDY "" CACHE STRING "Semi-colon list specifying clang-tidy command and arguments")
set(CMAKE_CXX_COMPILER_LAUNCHER "" CACHE STRING "Semi-colon list specifying command to wrap compiler invocations (e.g., ccache)")
set(USE_ARCH native CACHE STRING "Optimize for a specific processor architecture ('none' to disable)")

# This option is automatically handled by CMake.
# It makes add_library build a shared lib unless STATIC is explicitly specified.
# Putting this here is mostly just a placeholder so people know it's an option.
# Currently this is really only intended to change anything for the libgalois_shmem target.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libraries")

# Enable iss specific options; should be OFF in the general release; all guarded by USE_EXP
set(USE_EXP OFF CACHE BOOL "Use experimental features")
set(USE_DIST_EXP OFF CACHE BOOL "Use experimental features in distributed branch")
set(USE_HTM OFF CACHE BOOL "Use HTM")
set(USE_PROF OFF CACHE BOOL "Use profiling specific features")
set(USE_EMBEDDED_REVISION ON CACHE BOOL "Embed revision numbers")

# Enable cross compilation for specific architectures: For eg. -march=knl for knl architecture
set(USE_ARCH native CACHE STRING "Use specific architecture for cross compilation (Default : native)")

set(NUM_TEST_THREADS "" CACHE STRING "Maximum number of threads to use when running tests (default: number of physical cores)")

if(NOT NUM_TEST_THREADS)
  cmake_host_system_information(RESULT NUM_TEST_THREADS QUERY NUMBER_OF_PHYSICAL_CORES)
endif()
if(NUM_TEST_THREADS LESS_EQUAL 0)
  set(NUM_TEST_THREADS 1)
endif()

if(ENABLE_HETERO_GALOIS)
  set(USE_ARCH none)
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

  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Werror>")

  if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 7 AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 8)
    # GCC-{6,7} is a too imprecise
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-but-set-variable>")
  elseif(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 9)
    # Avoid warnings from boost::counting_iterator
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wno-deprecated-copy>")
    # Avoid warnings from openmpi
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wno-cast-function-type>")
  endif()

  # Removing unused parameters conflicts with readability rules
  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-parameter>")
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7)
    message(FATAL_ERROR "clang must be version 7 or higher. Found ${CMAKE_CXX_COMPILER_VERSION}.")
  endif()

  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Werror>")

  if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 10)
    # Avoid warnings from boost::counting_iterator
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wno-deprecated-copy>")
  endif()

  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-fcolor-diagnostics;-Wno-unused-parameter>")
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Werror>")

  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-parameter>")
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 19.0.1)
    message(FATAL_ERROR "icpc must be 19.0.1 or higher. Found ${CMAKE_CXX_COMPILER_VERSION}.")
  endif()

  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-wd68;-wd981;-wd383;-wd869;-wd2196;-wd279;-wd2504;-wd2943;-wd32013;-wd3373>")
endif()

if(CYGWIN)
  message(STATUS "On Cygwin")
  add_definitions(-U__STRICT_ANSI__)
endif()

# Enable architecture-specific optimizations
include(CheckArchFlags)
if(ARCH_FLAGS_FOUND)
  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:${ARCH_CXX_FLAGS}>")
  add_compile_options("$<$<COMPILE_LANGUAGE:C>:${ARCH_C_FLAGS}>")
  add_link_options(${ARCH_LINK_FLAGS})
endif()

if(USE_GPROF)
  add_compile_options(-pg)
  add_link_options(-pg)
endif(USE_GPROF)

if(USE_SANITIZER)
  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address;-fno-omit-frame-pointer;-fno-optimize-sibling-calls>")
endif()

if(CXX_CLANG_TIDY)
  set(CMAKE_CXX_CLANG_TIDY ${CXX_CLANG_TIDY} "-header-filter=.*${PROJECT_SOURCE_DIR}.*")
  # Ignore warning flags intended for the CXX program. This only works because
  # the two compilers we care about, clang and gcc, both understand
  # -Wno-unknown-warning-option.
  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wno-unknown-warning-option>")
endif()

###### Configure features ######

if(ENABLE_HETERO_GALOIS)
  set(ENABLE_DIST_GALOIS ON)
endif()

if(USE_EXP)
  set(USE_VTUNE ON)
  set(USE_PAPI ON)
  include_directories("libgalois/experimental/include")
  if(ENABLE_DIST_GALOIS)
    include_directories("libgluon/experimental/include")
  endif()

  find_package(OpenMP)
  if(OPENMP_FOUND)
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:${OpenMP_CXX_FLAGS}>")
    add_compile_options("$<$<COMPILE_LANGUAGE:C>:${OpenMP_CXX_FLAGS}>")
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
    if(CMAKE_CXX_COMPILER_ID STREQUAL "XL")
      add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-qtm;-qsmp=speculative>")
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
  find_package(VTune REQUIRED)
  include_directories(${VTune_INCLUDE_DIRS})
  add_definitions(-DGALOIS_USE_VTUNE)
endif()

if(USE_PAPI)
  if (PAPI_ROOT STREQUAL "")
    set(PAPI_ROOT /usr)
  endif()
  find_package(PAPI REQUIRED)
  include_directories(${PAPI_INCLUDE_DIRS})
  add_definitions(-DGALOIS_USE_PAPI)
endif()

find_package(Threads REQUIRED)

include(CheckMmap)

include(CheckHugePages)
if(NOT HAVE_HUGEPAGES AND USE_STRICT_CONFIG)
  message(FATAL_ERROR "Need huge pages")
endif()

if(USE_LONGJMP_ABORT)
  set(GALOIS_USE_LONGJMP_ABORT ON)
endif()

find_package(Boost 1.58.0 REQUIRED COMPONENTS serialization iostreams)

find_package(LLVM REQUIRED CONFIG)
if("${LLVM_PACKAGE_VERSION}" VERSION_LESS "7")
  MESSAGE(FATAL_ERROR "LLVM 7 or greater is required.")
endif()
if(NOT DEFINED LLVM_ENABLE_RTTI)
  MESSAGE(FATAL_ERROR "Could not determine if LLVM has RTTI enabled.")
endif()
if(NOT ${LLVM_ENABLE_RTTI})
  MESSAGE(FATAL_ERROR "Galois requires a build of LLVM that includes RTTI. Most package managers do this already, but if you built LLVM from source you need to configure it with `-DLLVM_ENABLE_RTTI=ON`")
endif()
target_include_directories(LLVMSupport INTERFACE ${LLVM_INCLUDE_DIRS})

include(CheckEndian)

###### Test Inputs ######

if(GRAPH_LOCATION)
  MESSAGE(STATUS "Using graph location ${GRAPH_LOCATION}")
  set(BASEINPUT "${GRAPH_LOCATION}/small_inputs")
elseif(EXISTS /net/ohm/export/iss/inputs)
  set(BASEINPUT /net/ohm/export/iss/inputs)
  MESSAGE(STATUS "Using graph location /net/ohm/export/iss/inputs")
else()
  set(BASEINPUT ${PROJECT_BINARY_DIR}/inputs/small_inputs)
  MESSAGE(STATUS "Use 'make input' to download inputs in the build directory")
endif()

###### Global Functions ######

function(add_test_scale type app)
  set(options NOT_QUICK)
  set(one_value_args)
  set(multi_value_args REQUIRES COMMAND_PREFIX)
  cmake_parse_arguments(X "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  set(threads)
  set(thr "${NUM_TEST_THREADS}")
  while (${thr} GREATER 1)
    list(APPEND threads ${thr})
    math(EXPR thr "${thr} / 2")
  endwhile()
  list(APPEND threads "1")

  foreach (thr ${threads})
    set(name run-${type}-${app}-${thr})
    add_test(NAME ${name} COMMAND ${app} ${X_UNPARSED_ARGUMENTS} -t ${thr})
    if (NOT ${X_NOT_QUICK})
      # Allow parallel tests
      set_tests_properties(${name}
        PROPERTIES ENVIRONMENT GALOIS_DO_NOT_BIND_THREADS=1 LABELS quick)
    endif()
  endforeach()
endfunction(add_test_scale)

function(add_test_unit)
  set(options)
  set(one_value_args ADD_TARGET)
  set(multi_value_args REQUIRES COMMAND_PREFIX)
  cmake_parse_arguments(X "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  if(NOT ${APP_ADD_TARGET})
    message(FATAL_ERROR "Missing required option ADD_TARGET to add_test_unit")
  endif()

  foreach(required ${X_REQUIRES})
    if(${${required}} MATCHES "TRUE")
    else()
      message(STATUS "NOT compiling ${name} (missing: ${required})")
      return()
    endif()
  endforeach()

  set(name ${X_ADD_TARGET})
  set(test_name unit-${name})

  add_executable(${test_name} ${name}.cpp)
  target_link_libraries(${test_name} galois_shmem lonestar)

  set(commandline ${X_COMMAND_PREFIX})
  list(APPEND commandline "$<TARGET_FILE:${test_name}>")
  list(APPEND commandline ${X_UNPARSED_ARGUMENTS})

  add_test(NAME ${test_name} COMMAND ${commandline})

  # Allow parallel tests
  set_tests_properties(${test_name}
    PROPERTIES
      ENVIRONMENT GALOIS_DO_NOT_BIND_THREADS=1
      LABELS quick
    )
endfunction()

###### Source finding ######

add_custom_target(lib)
add_custom_target(apps)

# Core libraries (lib)
add_subdirectory(libgalois)
if (ENABLE_DIST_GALOIS)
  find_package(MPI REQUIRED)
  add_subdirectory(libdist)
  add_subdirectory(libcusp)
  add_subdirectory(libgluon)
endif()
if (ENABLE_HETERO_GALOIS)
  enable_language(CUDA)
  string(REPLACE "." "" GENCODES ${CUDA_CAPABILITY})
  string(REPLACE "," ";" GENCODES ${GENCODES})
  foreach(GENCODE ${GENCODES})
    add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_${GENCODE},code=sm_${GENCODE}>")
  endforeach()

  add_subdirectory(libgpu)
endif()

# Applications (apps)
add_subdirectory(lonestar)
if (ENABLE_DIST_GALOIS)
  add_subdirectory(lonestardist)
endif()

add_subdirectory(scripts)
add_subdirectory(test)
add_subdirectory(inputs)
add_subdirectory(tools)

if(USE_DIST_EXP)
  add_subdirectory(lonestardist/experimental)
  add_subdirectory(lonestardist/experimental/resilience)
  add_subdirectory(lonestardist/experimental/on_demand)
endif()

if(USE_EXP)
  add_subdirectory(lonestar/experimental)
endif(USE_EXP)

###### Documentation ######

find_package(Doxygen)
if(DOXYGEN_FOUND)
  #TODO: double check the list of directories here
  set(DOXYFILE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/docs\" \"${CMAKE_CURRENT_SOURCE_DIR}/libcusp\" \"${CMAKE_CURRENT_SOURCE_DIR}/libdist\" \"${CMAKE_CURRENT_SOURCE_DIR}/libgalois\" \"${CMAKE_CURRENT_SOURCE_DIR}/libgluon")
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
