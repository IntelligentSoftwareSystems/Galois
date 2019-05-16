distApp(bc_level)
# TODO: fix app to include -exec and -maxIterations option
#testDistApp(bc_level rmat15 ${BASEINPUT}/scalefree/rmat15.gr -graphTranspose=${BASEINPUT}/scalefree/rmat15.tgr -singleSource)

distAppNoGPU(bc_mr)
# TODO: fix app to include -exec and -maxIterations option
#testDistApp(bc_mr rmat15 ${BASEINPUT}/scalefree/rmat15.gr -graphTranspose=${BASEINPUT}/scalefree/rmat15.tgr -numberOfSources=32)
