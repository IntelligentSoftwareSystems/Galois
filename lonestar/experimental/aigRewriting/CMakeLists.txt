file(GLOB Sources 
  subjectgraph/aig/*.cpp
  subjectgraph/cmxaig/*.cpp
  subjectgraph/cmxaig/nodes/*.cpp
  algorithms/*.cpp
  parsers/*.cpp
  writers/*.cpp
  misc/util/*.cpp
  functional/*.cpp
  factoring/*cpp
  covering/*.cpp
  mixer/*.cpp
  xxHash/xxhash.c
)

app(aigRewriting main.cpp ${Sources})

include_directories(./subjectgraph/aig)
include_directories(./subjectgraph/cmxaig)
include_directories(./subjectgraph/cmxaig/nodes)
include_directories(./algorithms)
include_directories(./parsers)
include_directories(.writers)
include_directories(./misc)
include_directories(./misc/util)
include_directories(./funcional)
include_directories(./factoring)
include_directories(./covering)
include_directories(./mixer)
include_directories(./xxHash)

