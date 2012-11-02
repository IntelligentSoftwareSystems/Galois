file (GLOB Sources 
./*.cpp
./common/*.cpp
./logic/*.cpp
)

app (DESunorderedSerial unordered/DESunorderedSerial.cpp ${Sources})
app (DESunordered unordered/DESunordered.cpp ${Sources})

app (DESorderedSerial ordered/DESorderedSerial.cpp ${Sources})
app (DESordered ordered/DESordered.cpp ${Sources})
app (DESorderedHand ordered/DESorderedHand.cpp ${Sources})
app (DESorderedHandNB ordered/DESorderedHandNB.cpp ${Sources})
app (DESorderedHandSet ordered/DESorderedHandSet.cpp ${Sources})


include_directories (.)
include_directories (./common)
include_directories (./logic)
include_directories (./ordered)
include_directories (./unordered)
