if(USE_EXP)
  include_directories(../../exp/apps/bfs .)
endif()

app(app-bfs bfs.cpp EXP_OPT)
app(app-diameter Diameter.cpp)
