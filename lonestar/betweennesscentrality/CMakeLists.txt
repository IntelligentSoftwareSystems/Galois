app(betweennesscentrality-outer BetweennessCentralityOuter.cpp)
app(bc-async BetweennessCentralityAsync.cpp)
app(bc-level BetweennessCentralityLevel.cpp)

add_test_scale(small betweennesscentrality-outer "${BASEINPUT}/scalefree/rmat10.gr")
