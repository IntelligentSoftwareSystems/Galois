file(GLOB Sources *.cpp)
add_definitions(-DGALOIS_CHANGE)
add_library(dmpruntime ${Sources})
