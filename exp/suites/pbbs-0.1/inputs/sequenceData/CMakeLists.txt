executable(randomSeq randomSeq.C)
executable(exptSeq exptSeq.C)
executable(almostSortedSeq almostSortedSeq.C)
executable(trigramSeq trigramSeq.C trigrams.C)
executable(addDataSeq addDataSeq.C)

function(trigramSeq base num)
  add_custom_command(OUTPUT ${input}/${base}
    COMMAND trigramSeq ${num} ${input}/${base}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/exp/pbbs/inputs/sequenceData/data
    DEPENDS trigramSeq)
  addInput(${base})
endfunction(trigramSeq)

function(trigramSeqPair base type num)
  trigramSeq(${base}.raw ${num})
  add_custom_command(OUTPUT ${input}/${base}
    COMMAND addDataSeq -t ${type} ${input}/${base}.raw ${input}/${base}
    DEPENDS addDataSeq ${input}/${base}.raw)
  addInput(${base})
endfunction(trigramSeqPair)

function(exptSeq base type num)
  add_custom_command(OUTPUT ${input}/${base}
    COMMAND exptSeq -t ${type} ${num} ${input}/${base}
    DEPENDS exptSeq)
  addInput(${base})
endfunction(exptSeq)

function(almostSortedSeq base type num)
  add_custom_command(OUTPUT ${input}/${base}
    COMMAND almostSortedSeq -t ${type} ${num} ${input}/${base}
    DEPENDS almostSortedSeq)
  addInput(${base})
endfunction(almostSortedSeq)

function(randomSeq base type num)
  add_custom_command(OUTPUT ${input}/${base}
    COMMAND randomSeq -t ${type} ${num} ${input}/${base}
    DEPENDS randomSeq)
  addInput(${base})
endfunction(randomSeq)

function(randomSeqPair base type num)
  randomSeq(${base}.raw ${type} ${num})
  add_custom_command(OUTPUT ${input}/${base}
    COMMAND addDataSeq -t ${type} ${input}/${base}.raw ${input}/${base}
    DEPENDS addDataSeq ${input}/${base}.raw)
  addInput(${base})
endfunction(randomSeqPair)

function(randomSeqRange base type range num)
  add_custom_command(OUTPUT ${input}/${base}
    COMMAND randomSeq -t ${type} -r ${range} ${num} ${input}/${base}
    DEPENDS randomSeq)
  addInput(${base})
endfunction(randomSeqRange)

function(randomSeqRangePair base type range num)
  randomSeqRange(${base}.raw ${type} ${range} ${num})
  add_custom_command(OUTPUT ${input}/${base}
    COMMAND addDataSeq -t ${type} ${input}/${base}.raw ${input}/${base}
    DEPENDS addDataSeq ${input}/${base}.raw)
  addInput(${base})
endfunction(randomSeqRangePair)
