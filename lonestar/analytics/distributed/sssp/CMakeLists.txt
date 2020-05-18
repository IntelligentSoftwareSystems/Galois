app_dist(sssp_push)
add_test_dist(sssp_push rmat15 ${BASEINPUT}/scalefree/rmat15.gr -graphTranspose=${BASEINPUT}/scalefree/transpose/rmat15.tgr)

app_dist(sssp_pull)
add_test_dist(sssp_pull rmat15 ${BASEINPUT}/scalefree/rmat15.gr -graphTranspose=${BASEINPUT}/scalefree/transpose/rmat15.tgr)
