app(boruvka Boruvka.cpp EXP_OPT)

add_test_scale(small1 boruvka "${BASEINPUT}/scalefree/rmat16-2e10-a=0.57-b=0.19-c=0.19-d=.05.gr")
add_test_scale(small2 boruvka "${BASEINPUT}/reference/structured/rome99.gr")
#add_test_scale(web boruvka "${BASEINPUT}/road/USA-road-d.USA.gr")
