app(preflowpush Preflowpush.cpp EXP_OPT)

add_test_scale(small1 preflowpush "${BASEINPUT}/reference/structured/rome99.gr" 0 100)
add_test_scale(small2 preflowpush "${BASEINPUT}/scalefree/rmat16-2e10-a=0.57-b=0.19-c=0.19-d=.05.gr" 0 100)
#add_test_scale(web preflowpush "${BASEINPUT}/random/r4-2e23.gr" 0 100)
