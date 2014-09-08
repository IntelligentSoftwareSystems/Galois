app(pagerank-iterate PageRankIterate.cpp)
app(pagerank-redistribute PageRankRedistribute.cpp)

include_directories(${CMAKE_SOURCE_DIR}/apps/pagerank)
app(PageRankChromatic PageRankChromatic.cpp)
