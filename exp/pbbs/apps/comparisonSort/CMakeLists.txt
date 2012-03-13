set(prefix comparisonSort)

trigramSeq(trigramSeq_100M 100000000)
exptSeq(exptSeq_100M_double double 100000000)
almostSortedSeq(almostSortedSeq_100M_double double 100000000)
randomSeq(randomSeq_100M_double double 100000000)

## 

include_directories(common)

add_subdirectory(quickSort)
add_subdirectory(sampleSort)
add_subdirectory(serialSort)
#add_subdirectory(stlParallelSort)
