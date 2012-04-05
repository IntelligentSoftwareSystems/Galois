set(prefix suffixArray)

find_program(BZCAT bzcat)

function(genData in out)
  add_custom_command(OUTPUT ${input}/${out} 
    COMMAND bzcat ${in} > ${input}/${out})
  addInput(${out})
endfunction(genData)

#genData("${CMAKE_CURRENT_SOURCE_DIR}/sequenceData/data/wikisamp.xml.bz2" winisamp.xml)

## 

include_directories(common)

add_subdirectory(serialKS)
