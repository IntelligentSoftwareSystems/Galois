if(USE_EXP)
  include_directories(../../../lonestar/connectedcomponents .)
endif()
app(instrumentedCC InstrumentedConnectedComponents.cpp)
