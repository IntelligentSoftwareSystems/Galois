app(pagerank-iterate PageRankIterate.cpp)
app(pagerank-redistribute PageRankRedistribute.cpp)

include_directories(${CMAKE_SOURCE_DIR}/lonestar/pagerank ${CMAKE_SOURCE_DIR}/apps/pagerank)
app(PageRankChromatic PageRankChromatic.cpp REQUIRES GCC_TUPLE_BUG_NOT_FOUND)
app(PageRankKDG PageRankKDG.cpp REQUIRES GCC_TUPLE_BUG_NOT_FOUND)

app(PageRankAlamere PageRankAlamere.cpp)

app(PageRankSet PageRankSet.cpp)

app(PagerankPull PageRank-pull.cpp)
