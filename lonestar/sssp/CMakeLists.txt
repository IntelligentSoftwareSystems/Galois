app(sssp SSSP.cpp)

add_test_scale(small1 sssp "${BASEINPUT}/reference/structured/rome99.gr" -delta 8)
add_test_scale(small2 sssp "${BASEINPUT}/scalefree/rmat10.gr" -delta 8)
#add_test_scale(web sssp "${BASEINPUT}/random/r4-2e26.gr" -delta 8)
