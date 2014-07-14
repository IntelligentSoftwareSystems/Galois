if(OPENMP_FOUND)
  add_executable(sparselu sparselu_for/sparselu.c sparselu_for/bots_common.c sparselu_for/bots_main.c)
endif()
app(gsparselu gsparselu_for/sparselu.cpp gsparselu_for/bots_common.c gsparselu_for/bots_main.c)
