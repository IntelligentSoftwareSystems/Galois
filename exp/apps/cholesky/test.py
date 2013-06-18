#!/usr/bin/env python

import numpy as np
import random
import cholesky

# Generate a test matrix

n = 10
density = 0.1

L = np.zeros((n,n))

for i in range(n):
    L[i][i] = random.randint(1,50)
    for j in range(i):
        if random.random() < density:
            L[i][j] = random.randint(1,50)
A = np.dot(L,L.T)

# Output matrix A
matrixfn = "matrix.tmp"
matrixfh = open(matrixfn, 'w')
for i in range(n):
    for j in range(n):
        if j != 0:
            matrixfh.write("\t")
        matrixfh.write(str(A[i][j]))
    matrixfh.write("\n")
matrixfh.close()

# Do symbolic elimination
filledfn = 'edgelist.tmp'
depfn = 'deplist.tmp'
grfn = 'edgelist.tmp.gr'
choleskyfn = 'choleskyedges.tmp'
cholesky.main(matrixfn, filledfn, depfn, choleskyfn,
              ordering=cholesky.ordering_sequential)

# ../../build/release/tools/graph-convert/graph-convert -doubleedgelist2gr %s %s

