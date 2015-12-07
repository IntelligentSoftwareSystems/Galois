file(GLOB Sources 
./*.cpp
./common/*.cpp
./logic/*.cpp
)

app(DESunorderedSerial unordered/DESunorderedSerial.cpp ${Sources})
app(DESunordered unordered/DESunordered.cpp ${Sources})

if (USE_EXP)
  app(DESorderedSerial ordered/DESorderedSerial.cpp ${Sources} EXP_OPT)
  app(DESordered ordered/DESordered.cpp ${Sources} EXP_OPT)
  app(DESorderedHand ordered/DESorderedHand.cpp ${Sources} EXP_OPT)
  app(DESorderedSpec ordered/DESorderedSpec.cpp ${Sources} EXP_OPT)
  app(DESlevelExec ordered/DESlevelExec.cpp ${Sources} EXP_OPT)
  app(DEStwoPhase ordered/DEStwoPhase.cpp ${Sources} EXP_OPT)
  app(DESorderedHandNB ordered/DESorderedHandNB.cpp ${Sources} EXP_OPT)
  app(DESorderedHandSet ordered/DESorderedHandSet.cpp ${Sources} EXP_OPT)
endif()


include_directories(.)
include_directories(./common)
include_directories(./logic)
include_directories(./ordered)
include_directories(./unordered)

add_test_scale(small DESunordered "${BASEINPUTS}/des/multTree6bit.net")
add_test_scale(web DESunordered "${BASEINPUT}/des/koggeStone64bit.net")
