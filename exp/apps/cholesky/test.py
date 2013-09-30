#!/usr/bin/env python

""" test - Test NumericCholesky on random (positive definite) matrices."""

import numpy as np
from diff_edgelists import diff
import cholesky
import random
import subprocess
import sys, os.path, tempfile, time

# Generate a test matrix
def randentry():
    """Generate a random entry for the matrix."""
    #return random.randint(1, 10)
    return random.random() + 0.5

def generate_matrix(size=10, density=0.1):
    """Generate a square symmetric positive definite matrix of given
    size and density. Discards highly conditioned matrices"""
    discarded = 0
    while True:
        # Generate matrix factor L, a lower triangular (positive) matrix
        factor = np.zeros((size, size))
        for i in range(size):
            factor[i][i] = randentry()
            for j in range(i):
                if random.random() < density:
                    factor[i][j] = randentry()
        # A=L*L'
        matrix = np.dot(factor, factor.T)

        # Verify that A is well-conditioned
        cond = np.linalg.cond(matrix)
        print "%d x %d [%f]: Condition number is %g" % \
            (size, size, density, cond)
        if cond < 1e10:
            if discarded > 0:
                time.sleep(1)
            return matrix, factor
        else:
            discarded += 1
            print "  -> Discarding matrix %d" % discarded

def generate_matrices(count=100, size_range=(2, 500)):
    for _ in range(count):
        # Random parameters for the matrix
        size = random.randint(size_range[0], size_range[1])
        density = random.random()*(8.0 if size > 20 else 0.5)/size + 0.0001
        assert(density < 0.5)
        print "Using size=%d, density=%f" % (size, density)

        cholmatfn = "matrix.L.tmp"
        matrixfn = "matrix.tmp"
    
        # Generate a random matrix
        matrix, factor = generate_matrix(size, density)
        cholesky.write_matrix(factor, cholmatfn)
        cholesky.write_matrix(matrix, matrixfn)
        yield (matrixfn, size, size, density)

def runcmd(cmd):
    """Run an external command and check the exit status."""
    print cmd
    status = subprocess.call(cmd)
    assert(status == 0)
    return status

def verify(items, check=lambda x: x == 0):
    """Verify that all elements of items are equal to correct_answer."""
    return check(items[0]) and len(set(items)) == 1

def run_test(app, matrixfn, status=None, do_verify=True):
    """Run a test on the given copy of NumericCholesky."""
    if not status:
        status = StatusBar(enabled=False)

    edgesfn = matrixfn + cholesky.EDGES_SUFFIX
    filledfn = matrixfn + cholesky.FILLED_SUFFIX
    depfn = matrixfn + cholesky.DEP_SUFFIX
    choleskyfn = matrixfn + cholesky.CHOLESKYEDGES_SUFFIX

    threadings = [1, 2, 4, 8]
    results = []
    status.push('')
    # Repeat for each ordering supported by cholesky.py
    for orderingname, ordering in cholesky.ORDERINGS.items():
        print "Using ordering: %s" % orderingname
        status.update('<%s>' % orderingname)
        status.push('cholesky.py')
        # Do symbolic elimination
        cholesky.do_cholesky(matrixfn, ordering=ordering)
        #status.update('graph-convert')
        #runcmd([graphconv, '-floatedgelist2gr', filledfn, grfn])

        # Test NumericCholesky, with several different numbers of threads
        for thr in threadings:
            if '/NumericCholesky' in app:
                status.update('NumericCholesky -t=%d' % thr)
                runcmd([app, '-t=%d' % thr, filledfn + '.gr', depfn])
            elif '/Cholesky' in app:
                status.update('Cholesky -t=%d' % thr)
                runcmd([app, '-ordering=%s' % orderingname, '-t=%d' % thr,
                        edgesfn + '.gr'])
            else:
                raise NotImplemented
            # Compare result
            status.update('Verifying...')
            results.append(diff((choleskyfn, 'choleskyedges.txt'), quick=True))
        status.pop() # Pop: Task for the given ordering

        # Check for any errors
        if not do_verify or verify(results):
            print "OK"
        else:
            print "Got bad results with %s on %s" % (ordering, threadings)
            print results
            raise Exception
    status.pop()     # Pop: ordering

def _main(argv):
    """Main function (entry point for command-line use)."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('app', metavar='APP',
                        help='Path to NumericCholesky binary')
    parser.add_argument('--iterations', type=int, default=100,
                        help='Number of iterations')
    parser.add_argument('--temp-dir', default=None,
                        help='Directory to store temporary files')
    parser.add_argument('--min-size', type=int, default=2,
                        help='Minimum matrix size to test')
    parser.add_argument('--max-size', type=int, default=500,
                        help='Maximum matrix size to test')
    parser.add_argument('--matrix', action='append',
                        help='Specific matrix to test')
    parser.add_argument('--no-verify', action='count',
                        help='Do not verify the result')
    args = parser.parse_args(argv)

    app = os.path.abspath(args.app)
    status = StatusBar()

    if args.matrix:
        args.matrix = [os.path.abspath(i) for i in args.matrix]

    # Change to a temporary directory
    if args.temp_dir:
        temp_dir = os.path.abspath(args.temp_dir)
    else:
        temp_dir = tempfile.mkdtemp()
    print 'TEMP_DIR = %s' % temp_dir
    os.chdir(temp_dir)

    # Get an iterator of the matrices to test
    if args.matrix:
        matrices = ((i,) for i in args.matrix)
        nmatrices = len(args.matrix)
    else:
        assert(args.min_size > 1 and args.min_size <= args.max_size)
        nmatrices = args.iterations
        matrices = generate_matrices(count=nmatrices,
                                     size_range=(args.min_size, args.max_size))

    # Test each matrix
    for i, matrix in enumerate(matrices):
        name = '%d/%d' % (i+1, nmatrices)
        if args.matrix:
            name += ' '+os.path.basename(matrix[0])
        print "="*79
        print name
        print "="*79
        status.push('[%s]' % name)
        if len(matrix) > 1:     # Display matrix info, if available
            status.push('[%dx%d/%.3f]' % matrix[1:])
        run_test(app, matrix[0], status=status, do_verify=not args.no_verify)
        if len(matrix) > 1:
            status.pop()        # Pop: matrix size
        status.pop()            # Pop: matrix name
    print "%d matrices OK" % (nmatrices)
    status.paint()

class StatusBar(object):
    """Display a status bar in the xterm titlebar"""

    def __init__(self, enabled=True):
        self.items = []
        self.enabled = False
        if not enabled:
            return
        supported_terms = ['xterm', 'rxvt', 'screen', 'cygwin']
        if 'TERM' in os.environ:
            for i in supported_terms:
                if os.environ['TERM'].startswith(i):
                    self.enabled = True
                    break

    def push(self, item):
        """Push an item onto the end of statusbar. Repaints the
        statusbar."""
        self.items.append(item)
        self.paint()

    def pop(self):
        """Remove the last item from the statusbar. Does not
        repaint the statusbar."""
        self.items.pop()

    def update(self, item):
        """Replace the last item of the statusbar with item. Repaints
        the statusbar."""
        self.pop()
        self.push(item)

    def paint(self):
        """Update the statusbar. Has no effect if not enabled
        (unsupported terminal type)."""
        if self.enabled:
            sys.stdout.write("\033]0;%s\007" % ' '.join(self.items))

if __name__ == '__main__':
    _main(sys.argv[1:])
