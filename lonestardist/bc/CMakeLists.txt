distApp(bc_level)
testDistSyncOnlyApp(bc_level rmat15 ${BASEINPUT}/scalefree/rmat15.gr -graphTranspose=${BASEINPUT}/scalefree/rmat15.tgr -singleSource)

distAppNoGPU(bc_mr)
testDistSyncOnlyNoGPUApp(bc_mr rmat15 ${BASEINPUT}/scalefree/rmat15.gr -graphTranspose=${BASEINPUT}/scalefree/rmat15.tgr -numOfSources=32)
