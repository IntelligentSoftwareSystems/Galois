app(bfs bfs.cpp)

add_test_scale(web bfs "${BASEINPUT}/random/r4-2e26.gr")
add_test_scale(small bfs "${BASEINPUT}/structured/rome99.gr")
