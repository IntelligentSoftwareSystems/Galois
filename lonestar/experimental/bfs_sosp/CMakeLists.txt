if(USE_EXP)
  include_directories(../../exp/apps/bfs .)
endif()

app(app-bfs bfs.cpp)
app(app-diameter Diameter.cpp)
