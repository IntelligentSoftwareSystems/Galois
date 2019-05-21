app(bfs bfs.cpp)

add_test_scale(small1 bfs "${BASEINPUT}/reference/structured/rome99.gr")
add_test_scale(small2 bfs "${BASEINPUT}/scalefree/rmat16-2e10-a=0.57-b=0.19-c=0.19-d=.05.gr")
#add_test_scale(web bfs "${BASEINPUT}/random/r4-2e26.gr")
