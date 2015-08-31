app(boruvka Boruvka.cpp)
app(boruvka-merge BoruvkaMerge.cpp)

add_test_scale(web boruvka "${BASEINPUT}/road/USA-road-d.USA.gr")
add_test_scale(small boruvka "${BASEINPUT}/structured/rome99.gr")
