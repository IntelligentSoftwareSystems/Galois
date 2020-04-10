file(GLOB Sources 
./*.cpp
./common/*.cpp
./logic/*.cpp
)

app(DESunorderedSerial unordered/DESunorderedSerial.cpp ${Sources})
app(DESunordered unordered/DESunordered.cpp ${Sources})

if (USE_EXP)
  app(DESorderedSerial ordered/DESorderedSerial.cpp ${Sources})
  app(DESordered ordered/DESordered.cpp ${Sources})
  app(DESorderedHand ordered/DESorderedHand.cpp ${Sources})
  app(DESorderedSpec ordered/DESorderedSpec.cpp ${Sources})
  app(DESlevelExec ordered/DESlevelExec.cpp ${Sources})
  app(DEStwoPhase ordered/DEStwoPhase.cpp ${Sources})
  app(DESorderedHandNB ordered/DESorderedHandNB.cpp ${Sources})
  app(DESorderedHandSet ordered/DESorderedHandSet.cpp ${Sources})
endif()


include_directories(.)
include_directories(./common)
include_directories(./logic)
include_directories(./ordered)
include_directories(./unordered)

add_test_scale(small DESunordered "${BASEINPUTS}/des/multTree6bit.net")
add_test_scale(web DESunordered "${BASEINPUT}/des/koggeStone64bit.net")
