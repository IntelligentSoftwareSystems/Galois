if(USE_EXP)
  include_directories(../../exp/apps/connectedcomponents .)
endif()
app(connectedcomponents)

add_test_scale(small connectedcomponents "${BASEINPUT}/scalefree/symmetric/rmat10.sgr")
#add_test_scale(web connectedcomponents "${BASEINPUT}/scalefree/randomized/symmetric/rmat16-2e25-a=0.57-b=0.19-c=0.19-d=.05.srgr")
