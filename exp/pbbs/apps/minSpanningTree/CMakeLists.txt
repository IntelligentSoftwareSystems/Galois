set(prefix minSpanningTree)

rmatWeighted(rMatGraph_WE_5_10000000 "YES" 10000000 50000000)
randWeighted(randLocalGraph_WE_5_10000000 "YES" 10000000 50000000)
gridWeighted(2Dgrid_WE_10000000 "YES" 2 10000000)

## 

include_directories(common)

add_subdirectory(parallelKruskal)
add_subdirectory(serialMST)
