find_package(SDL)
if(SDL_FOUND)
  app(drawgraph drawgraph.cpp EXTLIBS SDL)
endif()
