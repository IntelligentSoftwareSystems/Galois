app(sssp SSSP.cpp)

add_test_scale(small sssp "${BASEINPUT}/structured/rome99.gr")
add_test_scale(web sssp "${BASEINPUT}/random/r4-2e26.gr" -delta 8)
