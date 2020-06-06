app_dist(cc_push connected-components-push)
add_test_dist(connected-components-push-dist rmat15 ${BASEINPUT}/scalefree/symmetric/rmat15.sgr -symmetricGraph)

app_dist(cc_pull connected-components-pull)
add_test_dist(connected-components-pull-dist rmat15 ${BASEINPUT}/scalefree/symmetric/rmat15.sgr -symmetricGraph)
