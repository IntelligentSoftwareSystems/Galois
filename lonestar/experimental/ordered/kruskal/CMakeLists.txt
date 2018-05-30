app(KruskalSerial KruskalSerial.cpp)

if(USE_EXP) 
  app(KruskalDet KruskalDet.cpp EXP_OPT)
  app(KruskalHand KruskalHand.cpp EXP_OPT)
  app(KruskalLevelExec KruskalLevelExec.cpp EXP_OPT)
  app(KruskalOrdered KruskalOrdered.cpp EXP_OPT)
  app(KruskalSpec KruskalSpec.cpp EXP_OPT)
  app(KruskalIKDG KruskalIKDG.cpp EXP_OPT)
  app(KruskalStrictOBIM KruskalStrictOBIM.cpp EXP_OPT)

  add_test_scale(small KruskalHand "${BASEINPUT}/structured/rome99.gr")
  add_test_scale(web   KruskalHand "${BASEINPUT}/random/r4-2e24.gr" -maxRounds 600 -lowThresh 16 -preAlloc 32)

endif()
