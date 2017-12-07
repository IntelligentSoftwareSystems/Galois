file(GLOB Sources 
  subjectgraph/aig/*.cpp
  algorithms/*.cpp
  parsers/*.cpp
  writers/*.cpp
  misc/util/*.cpp
  functional/*.cpp
  xxHash/xxhash.c
)

app(aigRewriting main.cpp ${Sources})

include_directories(./subjectgraph/aig)
include_directories(./algorithms)
include_directories(./parsers)
include_directories(./writers)
include_directories(./misc)
include_directories(./misc/util)
include_directories(./funcional)
include_directories(./xxHash)

