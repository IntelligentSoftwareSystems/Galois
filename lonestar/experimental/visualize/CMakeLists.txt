find_package(SDL)
find_package(X11)
#This works when X11_FOUND isn't set, on some machines.  ugg.
app(drawgraph drawgraph.cpp REQUIRES SDL_FOUND EXTLIBS SDL X11)
