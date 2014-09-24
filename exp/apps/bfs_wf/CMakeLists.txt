app(BFSserialFIFO BFSserialFIFO.cpp)

app(BFSserialWavefront BFSserialWavefront.cpp REQUIRES GCC_TUPLE_BUG_NOT_FOUND)
app(BFSwavefrontLock BFSwavefrontLock.cpp REQUIRES GCC_TUPLE_BUG_NOT_FOUND)
app(BFSwavefrontNolock BFSwavefrontNolock.cpp REQUIRES GCC_TUPLE_BUG_NOT_FOUND)
app(BFSwavefrontCoupled BFSwavefrontCoupled.cpp REQUIRES GCC_TUPLE_BUG_NOT_FOUND)

app(SpecOrderedBFS SpecOrderedBFS.cpp)
app(TwoPhaseBFS TwoPhaseBFS.cpp)
app(LevelExecBFS LevelExecBFS.cpp)
app(BFSwavefrontBulkSync BFSwavefrontBulkSync.cpp)
app(BFSwavefrontBulkSyncEdge BFSwavefrontBulkSyncEdge.cpp)
# app(BFSwavefrontBag BFSwavefrontBag.cpp)
# app(BFSwavefrontEdge BFSwavefrontEdge.cpp)
