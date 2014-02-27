# Check if Fortran is possibly around before using enable_lanauge because
# enable_language(... OPTIONAL) does not fail gracefully if language is not
# found:
#  http://public.kitware.com/Bug/view.php?id=9220
set(Fortran_EXECUTABLE)
if(Fortran_EXECUTABLE)
  set(Fortran_FIND_QUIETLY TRUE)
endif()
find_program(Fortran_EXECUTABLE NAMES gfortran ifort g77 f77 g90 f90)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Fortran DEFAULT_MSG Fortran_EXECUTABLE)
if(FORTRAN_FOUND)
  set(Fortran_FOUND TRUE)
endif()
