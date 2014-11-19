find_program(PYTHON "python")
if(NOT PYTHON)
  message(STATUS "NOT compiling toolkit_ml python library: Could NOT find python")
  return()
endif()

execute_process(
  COMMAND ${PYTHON} -c "from Cython.Build import cythonize"
  ERROR_QUIET
  RESULT_VARIABLE HAVE_CYTHON)

if(NOT HAVE_CYTHON EQUAL "0")
  message(STATUS "NOT compiling toolkit_ml python library: Could NOT find cython")
  return()
endif()

# Generate setup.py
get_target_property(include_dir toolkit_ml INCLUDE_DIRECTORIES)
string(REPLACE ";" " -I" py_compile_args "-I${include_dir}")
get_target_property(location toolkit_ml LOCATION)
get_filename_component(dir "${location}" PATH)
set(py_link_args "-L${dir}")
configure_file(setup.py.in pylib/setup.py)

# Generate setup.cfg
set(wheel_py ${CMAKE_CURRENT_SOURCE_DIR}/buildscripts/wheelconfig.py)
execute_process(COMMAND ${PYTHON} ${wheel_py} platform OUTPUT_VARIABLE PLATFORM_TAG)
execute_process(COMMAND ${PYTHON} ${wheel_py} python OUTPUT_VARIABLE PYTHON_TAG)
configure_file(setup.cfg.in pylib/gsklearn/setup.cfg)

# Copy py scripts and build extensions
set(timestamp ${CMAKE_CURRENT_BINARY_DIR}/timestamp)
set(mod_py ${CMAKE_CURRENT_SOURCE_DIR}/buildscripts/lastmodtime.py)
add_custom_command(OUTPUT DUMMY ${timestamp}
  COMMAND ${PYTHON} ${mod_py} ${CMAKE_CURRENT_SOURCE_DIR}/pylib ${timestamp}.in
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${timestamp}.in ${timestamp}
  DEPENDS ${mod_py})
# First copy python scripts, then copy includes for cython build
add_custom_command(OUTPUT pylib.stamp 
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/pylib pylib
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/../include pylib/toolkit_ml
  COMMAND ${CMAKE_COMMAND} -E chdir pylib ${PYTHON} setup.py build_ext -i
  COMMAND ${CMAKE_COMMAND} -E touch pylib.stamp
  DEPENDS ${timestamp} toolkit_ml)
add_custom_target(gsklearn ALL DEPENDS DUMMY pylib.stamp)

install(CODE "execute_process(COMMAND ${PYTHON} pylib/setup.py install)")

function(makePyTest name)
  makeTest(${name} ${ARGN} COMMAND_PREFIX ${PYTHON})
  set_tests_properties(${name} PROPERTIES ENVIRONMENT
    PYTHONPATH=$ENV{PYTHONPATH}:${CMAKE_CURRENT_BINARY_DIR}/pylib)
endfunction()

makePyTest(pylib/gsklearn/tests/plot_separating_hyperplane.py)
