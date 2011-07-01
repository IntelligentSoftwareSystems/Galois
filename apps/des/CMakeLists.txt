file (GLOB Sources 
./*.cpp
./common/*.cpp
./common/circuitlib/*.cpp
)

add_executable(DESunorderedSerial unordered/DESunorderedSerial.cpp  ${Sources})
target_link_libraries(DESunorderedSerial galois) 
target_link_libraries(DESunorderedSerial ${CMAKE_THREAD_LIBS_INIT}) 
target_link_libraries(DESunorderedSerial hoard)


add_executable(DESunordered unordered/DESunordered.cpp  ${Sources})
target_link_libraries(DESunordered galois) 
target_link_libraries(DESunordered ${CMAKE_THREAD_LIBS_INIT}) 
target_link_libraries(DESunordered hoard)



include_directories (.)
include_directories (./common)
include_directories (./common/circuitlib)
include_directories (./unordered)
