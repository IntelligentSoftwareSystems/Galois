app(preflowpush Preflowpush.cpp EXP_OPT)

add_test_scale(small preflowpush "${BASEINPUT}/structured/rome99.gr" 0 100)
add_test_scale(web preflowpush "${BASEINPUT}/random/r4-2e23.gr" 0 100)
