include(CheckIncludeFile)
include(CheckCXXSourceCompiles)

macro(add_cxx_include result files)
  set(${result} "")
  foreach (file_name ${files})
     set(${result} "${${result}}#include<${file_name}>\n")
  endforeach()
endmacro(add_cxx_include files result)

function(check_type_exists type files variable)
  add_cxx_include(includes "${files}")
  CHECK_CXX_SOURCE_COMPILES("
    ${includes} ${type} typeVar;
    int main() {
        return 0;
    }
    " ${variable})
endfunction()

check_include_file(sys/types.h HAVE_SYS_TYPES_H)
check_include_file(inttypes.h HAVE_INTTYPES_H)
check_include_file(stdint.h HAVE_STDINT_H)

set(headers "")
if (HAVE_SYS_TYPES_H)
  set(headers ${headers} "sys/types.h")
endif()

if (HAVE_INTTYPES_H)
  set(headers ${headers} "inttypes.h")
endif()

if (HAVE_STDINT_H)
  set(headers ${headers} "stdint.h")
endif()

check_type_exists(int64_t "${headers}" HAVE_INT64_T)
check_type_exists(uint64_t "${headers}" HAVE_UINT64_T)
check_type_exists(u_int64_t "${headers}" HAVE_U_INT64_T)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/include/llvm/Support/DataTypes.h.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/include/llvm/Support/DataTypes.h)

include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
