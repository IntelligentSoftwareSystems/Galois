#!/usr/bin/env python

import numpy as np
import random
import cholesky
import subprocess
import sys, os.path, tempfile

# Generate a test matrix
def generate_example(size=10, density=0.1):
    L = np.zeros((size, size))

    for i in range(size):
        L[i][i] = random.randint(1,50)
        for j in range(i):
            if random.random() < density:
                L[i][j] = random.randint(1,50)
    A = np.dot(L,L.T)
    return A, L

def compare_edgelists(*fns):
    fhs = [open(i) for i in fns]
    for lines in zip(*fhs):
        line = lines[0].split()
        a = int(line[0])
        b = int(line[1])
        weight = float(line[2])
        for line in lines:
            line = line.split()
            assert(int(line[0]) == a and int(line[1]) == b and float(line[2]) == weight)

def main(bin):
    A, L = generate_example()
    # Do symbolic elimination
    matrixfn = "matrix.tmp"
    filledfn = 'edgelist.tmp'
    depfn = 'deplist.tmp'
    grfn = 'edgelist.tmp.gr'
    choleskyfn = 'choleskyedges.tmp'

    cholesky.write_matrix(A, matrixfn)
    cholesky.main(matrixfn, filledfn, depfn, choleskyfn,
                  ordering=cholesky.ordering_sequential)

    # Convert the graph to Galois binary format
    subprocess.call([os.path.join(bin, 'tools/graph-convert/graph-convert'),
                     '-floatedgelist2gr', filledfn, grfn])

    # Perform numeric factorization
    subprocess.call([os.path.join(bin, 'exp/apps/cholesky/NumericCholesky'),
                     '-t=8', grfn, depfn])

    # Compare result
    compare_edgelists(choleskyfn, 'choleskyedges.txt')
    print "OK"

if __name__ == '__main__':
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print "Usage: %s <build-tree> [tmp-dir]" % sys.argv[0]
        print "Example: %s ~/Galois/build/default /tmp/cholesky"
        sys.exit(1)
    BIN_DIR = os.path.abspath(sys.argv[1])

    try:
        TEMP_DIR = os.path.abspath(sys.argv[2])
    except IndexError:
        TEMP_DIR = tempfile.mkdtemp()
    print 'TEMP_DIR = %s' % TEMP_DIR
    os.chdir(TEMP_DIR)
    main(BIN_DIR)
