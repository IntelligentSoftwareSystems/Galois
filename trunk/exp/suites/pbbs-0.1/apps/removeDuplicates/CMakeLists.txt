set(prefix removeDuplicates)

randomSeq(randomSeq_10M_int int 10000000)
randomSeqRange(randomSeq_10M_100K_int int 100000 10000000)
exptSeq(exptSeq_10M_int int 10000000)
trigramSeq(trigramSeq_10M 10000000)
trigramSeqPair(trigramSeq_10M_pair_int int 10000000)

## 

include_directories(common)

add_subdirectory(deterministicHash)
add_subdirectory(serialHash)
