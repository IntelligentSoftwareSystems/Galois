#!/usr/bin/env/python
from cholesky import load_matrix, write_edgelist, write_gr
from sys import argv, exit
if len(argv) != 3:
    print "Usage: %s <input matrix> <output gr>" % argv[0]
    exit(1)
graph = load_matrix(argv[1]).to_directed()
write_edgelist(graph)
write_gr(graph, argv[2])
