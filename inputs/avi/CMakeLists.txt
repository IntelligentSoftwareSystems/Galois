find_program(WGET wget)
find_program(GUNZIP gunzip)

function(dlinput base url)
  add_custom_command(OUTPUT ${base}.gz
    COMMAND ${WGET} -O ${base}.gz ${url})
endfunction(dlinput)

dlinput(10x10_42k.NEU.gz http://iss.ices.utexas.edu/projects/galois/downloads/inputs/avi/10x10_42k.NEU.gz)

add_custom_target(more-avi-inputs DEPENDS 10x10_42k.NEU.gz)
add_dependencies(more-inputs more-avi-inputs)
