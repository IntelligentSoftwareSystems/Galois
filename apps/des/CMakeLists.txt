file (GLOB Sources 
./*.cpp
./common/*.cpp
)

function (DEStarget name) 
  add_executable (${name} unordered/${name}.cpp ${Sources})
  linkRuntime (${name})
endfunction (DEStarget)

DEStarget (DESunorderedSerial)

DEStarget (DESunordered)

DEStarget (DESunorderedParaMeter)


include_directories (.)
include_directories (./common)
include_directories (./common/circuitlib)
include_directories (./unordered)
