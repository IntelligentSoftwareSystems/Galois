app(betweennesscentrality-outer BetweennessCentralityOuter.cpp)

add_test_scale(web betweennesscentrality-outer "${BASEINPUT}/scalefree/rmat8-2e14.gr")
add_test_scale(small betweennesscentrality-outer "${BASEINPUT}/structured/torus5.gr")
