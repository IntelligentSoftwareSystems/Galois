file (GLOB Sources 
./*.cpp
./common/*.cpp
./common/circuitlib/*.cpp
)

add_executable(DESunorderedSerial unordered/DESunorderedSerial.cpp  ${Sources})
linkRuntime (DESunorderedSerial)


add_executable(DESunordered unordered/DESunordered.cpp  ${Sources})
linkRuntime (DESunordered)


include_directories (.)
include_directories (./common)
include_directories (./common/circuitlib)
include_directories (./unordered)
