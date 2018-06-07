app(boruvka Boruvka.cpp EXP_OPT)

add_test_scale(web boruvka "${BASEINPUT}/road/USA-road-d.USA.gr")
add_test_scale(small boruvka "${BASEINPUT}/structured/rome99.gr")
