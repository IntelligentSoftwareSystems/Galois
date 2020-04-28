app_dist(bc_level)
add_test_dist(bc_level rmat15 NO_ASYNC ${BASEINPUT}/scalefree/rmat15.gr -graphTranspose=${BASEINPUT}/scalefree/transpose/rmat15.tgr -singleSource)

app_dist(bc_mr NO_GPU)
add_test_dist(bc_mr rmat15 NO_ASYNC NO_GPU ${BASEINPUT}/scalefree/rmat15.gr -graphTranspose=${BASEINPUT}/scalefree/transpose/rmat15.tgr -numOfSources=128 -numRoundSources=32)
#add_test_dist(bc_mr rmat15all NO_ASYNC NO_GPU NOT_QUICK ${BASEINPUT}/scalefree/rmat15.gr -graphTranspose=${BASEINPUT}/scalefree/transpose/rmat15.tgr -numRoundSources=4096)
