set(prefix integerSort)

randomSeqRangePair(randomSeq_100M_256_int_pair_int int 256 100000000)
randomSeqPair(randomSeq_100M_int_pair_int int 100000000)
randomSeq(randomSeq_100M_int int 100000000)
exptSeq(exptSeq_100M_int int 100000000)

## 

include_directories(common)

add_subdirectory(blockRadixSort)
add_subdirectory(serialRadixSort)
