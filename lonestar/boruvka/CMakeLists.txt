app(boruvka Boruvka.cpp)

add_test_scale(small1 boruvka "${BASEINPUT}/scalefree/rmat10.gr")
add_test_scale(small2 boruvka "${BASEINPUT}/reference/structured/rome99.gr")
#add_test_scale(web boruvka "${BASEINPUT}/road/USA-road-d.USA.gr")
