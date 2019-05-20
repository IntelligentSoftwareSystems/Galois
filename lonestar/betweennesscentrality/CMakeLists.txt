app(betweennesscentrality-outer BetweennessCentralityOuter.cpp)
app(bc-async BetweennessCentralityAsync.cpp)
app(bc-level BetweennessCentralityLevel.cpp)

add_test_scale(small betweennesscentrality-outer "${BASEINPUT}/scalefree/rmat16-2e10-a=0.57-b=0.19-c=0.19-d=.05.gr")
#add_test_scale(web betweennesscentrality-outer "${BASEINPUT}/scalefree/rmat8-2e14.gr")
