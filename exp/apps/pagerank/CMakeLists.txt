app(pagerank-iterate PageRankIterate.cpp)
app(pagerank-redistribute PageRankRedistribute.cpp)

include_directories(${CMAKE_SOURCE_DIR}/apps/pagerank)
app(PageRankChromatic PageRankChromatic.cpp)
app(PageRankAddRemove PageRankAddRemove.cpp)
app(PageRankIKDG PageRankIKDG.cpp)
app(PageRankKDG PageRankKDG.cpp)
