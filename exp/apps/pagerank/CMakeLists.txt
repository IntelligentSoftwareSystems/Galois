app(pagerank-iterate PageRankIterate.cpp)
app(pagerank-redistribute PageRankRedistribute.cpp)

include_directories(${CMAKE_SOURCE_DIR}/lonestar/pagerank)
app(PageRankChromatic PageRankChromatic.cpp REQUIRES GCC_TUPLE_BUG_NOT_FOUND)
app(PageRankAddRemove PageRankAddRemove.cpp)
app(PageRankIKDG PageRankIKDG.cpp)
app(PageRankKDG PageRankKDG.cpp REQUIRES GCC_TUPLE_BUG_NOT_FOUND)

app(PageRankAlamere PageRankAlamere.cpp REQUIRES GCC_TUPLE_BUG_NOT_FOUND)

app(PageRankSet PageRankSet.cpp)
