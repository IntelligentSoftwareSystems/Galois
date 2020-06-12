# Galois: taken from:
#   https://github.com/llvm/llvm-project/blob/master/llvm/cmake/modules/HandleLLVMOptions.cmake 

include(CheckCCompilerFlag)
include(CheckCXXCompilerFlag)

string(TOUPPER "${CMAKE_BUILD_TYPE}" uppercase_CMAKE_BUILD_TYPE)

if(NOT GALOIS_USE_SANITIZER)
  return()
endif()

function(append value)
  foreach(variable ${ARGN})
    set(${variable} "${${variable}} ${value}" PARENT_SCOPE)
  endforeach(variable)
endfunction()

function(append_if condition value)
  if (${condition})
    foreach(variable ${ARGN})
      set(${variable} "${${variable}} ${value}" PARENT_SCOPE)
    endforeach(variable)
  endif()
endfunction()

macro(add_flag_if_supported flag name)
  check_c_compiler_flag("-Werror ${flag}" "C_SUPPORTS_${name}")
  append_if("C_SUPPORTS_${name}" "${flag}" CMAKE_C_FLAGS)
  check_cxx_compiler_flag("-Werror ${flag}" "CXX_SUPPORTS_${name}")
  append_if("CXX_SUPPORTS_${name}" "${flag}" CMAKE_CXX_FLAGS)
endmacro()

macro(append_common_sanitizer_flags)
  # Append -fno-omit-frame-pointer and turn on debug info to get better
  # stack traces.
  add_flag_if_supported("-fno-omit-frame-pointer" FNO_OMIT_FRAME_POINTER)
  if (NOT uppercase_CMAKE_BUILD_TYPE STREQUAL "DEBUG" AND
      NOT uppercase_CMAKE_BUILD_TYPE STREQUAL "RELWITHDEBINFO")
    add_flag_if_supported("-gline-tables-only" GLINE_TABLES_ONLY)
  endif()
  # Use -O1 even in debug mode, otherwise sanitizers slowdown is too large.
  if (uppercase_CMAKE_BUILD_TYPE STREQUAL "DEBUG")
    add_flag_if_supported("-O1" O1)
  endif()
endmacro()

if (GALOIS_USE_SANITIZER STREQUAL "Address")
  append_common_sanitizer_flags()
  append("-fsanitize=address" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
elseif (GALOIS_USE_SANITIZER STREQUAL "HWAddress")
  append_common_sanitizer_flags()
  append("-fsanitize=hwaddress" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
elseif (GALOIS_USE_SANITIZER MATCHES "Memory(WithOrigins)?")
  append_common_sanitizer_flags()
  append("-fsanitize=memory" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
  if(GALOIS_USE_SANITIZER STREQUAL "MemoryWithOrigins")
    append("-fsanitize-memory-track-origins" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
  endif()
elseif (GALOIS_USE_SANITIZER STREQUAL "Undefined")
  append_common_sanitizer_flags()
  append("-fsanitize=undefined -fno-sanitize-recover=all"
          CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
elseif (GALOIS_USE_SANITIZER STREQUAL "Thread")
  append_common_sanitizer_flags()
  append("-fsanitize=thread" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
elseif (GALOIS_USE_SANITIZER STREQUAL "DataFlow")
  append("-fsanitize=dataflow" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
elseif (GALOIS_USE_SANITIZER STREQUAL "Address;Undefined" OR
        GALOIS_USE_SANITIZER STREQUAL "Undefined;Address")
  append_common_sanitizer_flags()
  append("-fsanitize=address,undefined -fno-sanitize-recover=all"
          CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
elseif (GALOIS_USE_SANITIZER STREQUAL "Leaks")
  append_common_sanitizer_flags()
  append("-fsanitize=leak" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
else()
  message(FATAL_ERROR "Unsupported value of GALOIS_USE_SANITIZER: ${GALOIS_USE_SANITIZER}")
endif()

if (GALOIS_USE_SANITIZER MATCHES "(Undefined;)?Address(;Undefined)?")
  add_flag_if_supported("-fsanitize-address-use-after-scope"
                        FSANITIZE_USE_AFTER_SCOPE_FLAG)
endif()

if (GALOIS_USE_SANITIZE_COVERAGE)
  append("-fsanitize=fuzzer-no-link" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
endif()

if (GALOIS_USE_SANITIZER MATCHES ".*Undefined.*")
  set(BLACKLIST_CONFIGURE_FILE "${PROJECT_SOURCE_DIR}/config/sanitizers/ubsan_blacklist.txt.in")
  if (EXISTS "${BLACKLIST_CONFIGURE_FILE}")
    set(BLACKLIST_FILE "${PROJECT_BINARY_DIR}/config/sanitizers/ubsan_blacklist.txt")
    configure_file("${BLACKLIST_CONFIGURE_FILE}" "${BLACKLIST_FILE}")
    append("-fsanitize-blacklist=${BLACKLIST_FILE}"
           CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
  endif()
endif()
