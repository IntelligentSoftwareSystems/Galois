app(pagerank-iterate PageRankIterate.cpp)
app(pagerank-redistribute PageRankRedistribute.cpp)

include_directories(${CMAKE_SOURCE_DIR}/lonestar/pagerank ${CMAKE_SOURCE_DIR}/apps/pagerank)
app(PageRankChromatic PageRankChromatic.cpp REQUIRES GCC_TUPLE_BUG_NOT_FOUND EXP_OPT)
#app(PageRankAddRemove PageRankAddRemove.cpp EXP_OPT)
#app(PageRankIKDG PageRankIKDG.cpp EXP_OPT)
app(PageRankKDG PageRankKDG.cpp REQUIRES GCC_TUPLE_BUG_NOT_FOUND EXP_OPT)

app(PageRankAlamere PageRankAlamere.cpp EXP_OPT)

app(PageRankSet PageRankSet.cpp)
