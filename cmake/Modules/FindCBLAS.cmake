# Copyright 2009-2011 The VOTCA Development Team (http://www.votca.org)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

#! \file
#! \ingroup FindPackage
#! \brief Find CBLAS
#!
#! Find the native CBLAS headers and libraries.
#!
#! - `CBLAS_LIBRARIES`    - List of libraries when using cblas.
#! - `CBLAS_INCLUDE_DIRS` - List of include directories
#! - `CBLAS_FOUND`        - True if cblas found.
#!
#! Cblas can be provided by libblas (Ubuntu), cblas or gslcblas, it will be searched for in
#! this order.

include(LibFindMacros)

if (UNIX)
  find_package(PkgConfig QUIET)
  pkg_check_modules(CBLAS_PKGCONF QUIET cblas)
endif()

if (NOT CBLAS_FOUND)

if(CBLAS_PKGCONF_FOUND)

foreach(NEW_CBLAS_LIB ${CBLAS_PKGCONF_LIBRARIES})
  find_library(LIB_${NEW_CBLAS_LIB} ${NEW_CBLAS_LIB} HINTS ${CBLAS_PKGCONF_LIBRARY_DIRS})
  if(NOT LIB_${NEW_CBLAS_LIB})
    message(FATAL_ERROR "Could not find ${NEW_CBLAS_LIB} where pkgconfig said it is: ${CBLAS_PKGCONF_LIBRARY_DIRS}")
  else(NOT LIB_${NEW_CBLAS_LIB})
    message(STATUS "Found ${LIB_${NEW_CBLAS_LIB}}.")
  endif(NOT LIB_${NEW_CBLAS_LIB})
  set(CBLAS_LIBRARY ${CBLAS_LIBRARY} ${LIB_${NEW_CBLAS_LIB}})
endforeach(NEW_CBLAS_LIB)

else(CBLAS_PKGCONF_FOUND)

set(CBLAS_HINT_PATH $ENV{CBLASDIR}/lib $ENV{CBLASDIR}/lib64 $ENV{UIBK_GSL_LIB})

# Check if libblas provides cblas (Ubuntu)
find_library(BLAS_LIBRARY NAMES blas PATHS ${CBLAS_HINT_PATH})
if(BLAS_LIBRARY)
  include(CheckSymbolExists)
  set(CMAKE_REQUIRED_LIBRARIES ${BLAS_LIBRARY})
  check_symbol_exists(cblas_scopy "cblas.h" BLAS_HAS_CBLAS)
endif(BLAS_LIBRARY)

set(CBLAS_CANDIDATES cblas gslcblas)
if(BLAS_HAS_CBLAS)
  message(STATUS "libblas provides cblas.")
  set(CBLAS_CANDIDATES blas ${CBLAS_CANDIDATES})
endif(BLAS_HAS_CBLAS)

find_library(CBLAS_LIBRARY
  NAMES ${CBLAS_CANDIDATES}
  PATHS ${CBLAS_HINT_PATH}
)
endif(CBLAS_PKGCONF_FOUND)

if("${CBLAS_LIBRARY}" MATCHES gslcblas)
  set(CBLAS_INCLUDE_CANDIDATE gsl/gsl_cblas.h)
else("${CBLAS_LIBRARY}" MATCHES gslcblas)
  set(CBLAS_INCLUDE_CANDIDATE cblas.h)
endif("${CBLAS_LIBRARY}" MATCHES gslcblas)

find_path(CBLAS_INCLUDE_DIR ${CBLAS_INCLUDE_CANDIDATE} HINTS ${CBLAS_PKGCONF_INCLUDE_DIRS} $ENV{CBLASDIR}/include $ENV{UIBK_GSL_INC})

# Set the include dir variables and the libraries and let libfind_process do the rest.
# NOTE: Singular variables for this library, plural for libraries this this lib depends on.
set(CBLAS_PROCESS_INCLUDES CBLAS_INCLUDE_DIR)
set(CBLAS_PROCESS_LIBS CBLAS_LIBRARY)
libfind_process(CBLAS)
message(STATUS "Using '${CBLAS_LIBRARIES}' for cblas.")

endif(NOT CBLAS_FOUND)
