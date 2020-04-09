app(bfs bfs.cpp)

add_test_scale(small1 bfs "${BASEINPUT}/reference/structured/rome99.gr")
add_test_scale(small2 bfs "${BASEINPUT}/scalefree/rmat10.gr")
