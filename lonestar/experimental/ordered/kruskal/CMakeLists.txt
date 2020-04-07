app(KruskalSerial KruskalSerial.cpp)

if(USE_EXP) 
  app(KruskalDet KruskalDet.cpp)
  app(KruskalHand KruskalHand.cpp)
  app(KruskalLevelExec KruskalLevelExec.cpp)
  app(KruskalOrdered KruskalOrdered.cpp)
  app(KruskalSpec KruskalSpec.cpp)
  app(KruskalIKDG KruskalIKDG.cpp)
  app(KruskalStrictOBIM KruskalStrictOBIM.cpp)

  add_test_scale(small KruskalHand "${BASEINPUT}/structured/rome99.gr")
  add_test_scale(web   KruskalHand "${BASEINPUT}/random/r4-2e24.gr" -maxRounds 600 -lowThresh 16 -preAlloc 32)
endif()
