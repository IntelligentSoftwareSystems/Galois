app_dist(sssp_push sssp-push)
add_test_dist(sssp-push-dist rmat15 ${BASEINPUT}/scalefree/rmat15.gr -graphTranspose=${BASEINPUT}/scalefree/transpose/rmat15.tgr)

app_dist(sssp_pull sssp-pull)
add_test_dist(sssp-pull-dist rmat15 ${BASEINPUT}/scalefree/rmat15.gr -graphTranspose=${BASEINPUT}/scalefree/transpose/rmat15.tgr)
