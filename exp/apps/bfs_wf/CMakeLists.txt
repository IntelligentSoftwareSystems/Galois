app(BFSserialFIFO BFSserialFIFO.cpp EXP_OPT)

app(BFSserialWavefront BFSserialWavefront.cpp REQUIRES GCC_TUPLE_BUG_NOT_FOUND EXP_OPT)
app(BFSwavefrontLock BFSwavefrontLock.cpp REQUIRES GCC_TUPLE_BUG_NOT_FOUND EXP_OPT)
app(BFSwavefrontNolock BFSwavefrontNolock.cpp REQUIRES GCC_TUPLE_BUG_NOT_FOUND EXP_OPT)

app(SpecOrderedBFS SpecOrderedBFS.cpp EXP_OPT)
app(TwoPhaseBFS TwoPhaseBFS.cpp EXP_OPT)
app(LevelExecBFS LevelExecBFS.cpp EXP_OPT)
app(BFSwavefrontBulkSync BFSwavefrontBulkSync.cpp EXP_OPT)
app(BFSwavefrontCoupled BFSwavefrontCoupled.cpp EXP_OPT)
#app(BFSwavefrontBulkSyncEdge BFSwavefrontBulkSyncEdge.cpp EXP_OPT)
# app(BFSwavefrontBag BFSwavefrontBag.cpp)
# app(BFSwavefrontEdge BFSwavefrontEdge.cpp)
