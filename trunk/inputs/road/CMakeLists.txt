find_program(WGET wget)
find_program(GUNZIP gunzip)

function(dlinput base url)
  add_custom_command(OUTPUT ${base}.gz
    COMMAND ${WGET} -O ${base}.gz ${url})
  add_custom_command(OUTPUT ${base}.dimacs
    COMMAND ${GUNZIP} -c ${base}.gz > ${base}.dimacs
    DEPENDS ${base}.gz)
  add_custom_command(OUTPUT ${base}.gr
    COMMAND graph-convert -dimacs2gr ${base}.dimacs ${base}.gr
    DEPENDS ${base}.dimacs graph-convert)
endfunction(dlinput)

dlinput(USA-road-d.USA http://www.dis.uniroma1.it/~challenge9/data/USA-road-d/USA-road-d.USA.gr.gz)
dlinput(USA-road-d.W http://www.dis.uniroma1.it/~challenge9/data/USA-road-d/USA-road-d.W.gr.gz)
dlinput(USA-road-d.NY http://www.dis.uniroma1.it/~challenge9/data/USA-road-d/USA-road-d.NY.gr.gz)

add_custom_target(more-road-graphs DEPENDS USA-road-d.USA.gr USA-road-d.W.gr USA-road-d.NY.gr)
add_dependencies(more-inputs more-road-graphs)
