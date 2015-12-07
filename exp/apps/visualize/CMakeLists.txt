find_package(SDL)
app(drawgraph drawgraph.cpp REQUIRES SDL_FOUND EXTLIBS SDL)
