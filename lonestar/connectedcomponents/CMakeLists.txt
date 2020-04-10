if(USE_EXP)
  include_directories(../../exp/apps/connectedcomponents .)
endif()
app(connectedcomponents ConnectedComponents.cpp)

add_test_scale(small connectedcomponents "${BASEINPUT}/scalefree/symmetric/rmat10.sgr")
