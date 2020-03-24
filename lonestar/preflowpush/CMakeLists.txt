app(preflowpush Preflowpush.cpp)

add_test_scale(small1 preflowpush "${BASEINPUT}/reference/structured/torus5.gr" 0 10)
