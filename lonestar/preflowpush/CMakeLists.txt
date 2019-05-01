app(preflowpush Preflowpush.cpp EXP_OPT)

add_test_scale(small1 preflowpush "${BASEINPUT}/reference/structured/torus5.gr" 0 10)
