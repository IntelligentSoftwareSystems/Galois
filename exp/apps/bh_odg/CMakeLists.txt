if (CMAKE_COMPILER_IS_GNUCC)
  execute_process(COMMAND ${CMAKE_CXX_COMPILER} -dumpversion
                  OUTPUT_VARIABLE GCC_VERSION)
  if(GCC_VERSION VERSION_GREATER 4.8 OR GCC_VERSION VERSION_EQUAL 4.8)
    app (barneshut_odg BarneshutODG.cpp)
  endif()
else()
  app (barneshut_odg BarneshutODG.cpp)
endif()
