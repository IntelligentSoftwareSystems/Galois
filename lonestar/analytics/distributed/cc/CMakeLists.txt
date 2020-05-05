app_dist(cc_push)
add_test_dist(cc_push rmat15 ${BASEINPUT}/scalefree/symmetric/rmat15.sgr -symmetricGraph)

app_dist(cc_pull)
add_test_dist(cc_pull rmat15 ${BASEINPUT}/scalefree/symmetric/rmat15.sgr -symmetricGraph)
