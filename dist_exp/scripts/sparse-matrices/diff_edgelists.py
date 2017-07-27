#!/usr/bin/env python

"""diff_edgelists - Display differences between two edgelist files."""

import sys
from math import isnan, isinf
from itertools import izip_longest

# If within this tolerance, difference is not an error. (Surely five
# nines of accuracy is good enough for *relative* error.)
CLOSE_ENOUGH = 1e-5

class DiffEdgelists(object):
    """Compare edgelist files and display differences."""

    def __init__(self, symmetric=True, quick=False, verbose=True):
        self.symmetric = symmetric
        self.quick = quick
        self.verbose = verbose
        self.reset()

    def reset(self):
        """Reset all counters to zero."""
        # Counters
        self.nfiles = 0    # Number of files read
        self.nedges = 0    # Total number of edges
        self.ndups = 0     # Edges duplicated within one input file
        self.ninvalid = 0  # Edges that are invalid (NaN or infinite)
        self.ndiffs = 0    # Edges with differences > CLOSE_ENOUGH
        self.nexact = 0    # Edges with no measurable difference
        self.nxzeros = 0   # Explicit zeros not in all input files
        self.nmissing = 0  # Nonzero edges missing in some input files

        # Maximums
        self.maxabsdiff = 0     # Maximum absolute difference
        self.maxreldiff = 0     # Maximum relative difference
        self.maxqueue = 0       # Maximum size of queue

        # Verify reset was correct
        assert self.is_success()

    def _diff_edge(self, index, edge, incremental=True):
        """Compare two edges and return True if they are the same, False if a
        difference was found, or None if the edge was not found in
        every file and an incremental diff was requested.

        """
        if incremental and None in edge:
            # Edge is not complete
            return None
        eset = set(edge)
        esetlen = len(eset)
        if esetlen == 1:
            # Every version of edge is the same, no error
            self.nexact += 1
            return True
        elif not incremental and None in eset:
            if esetlen == 2:
                if float(0) in eset:
                    # An explicit zero that is not in every input file
                    self.nxzeros += 1
                    return True
                else:
                    self.nmissing += 1
                    if self.verbose >= 2:
                        print "%d %d" % index, edge
                    return False
            eset.remove(None)

        # Compute differences
        emin = min(eset)
        emax = max(eset)
        absdiff = emax-emin
        reldiff = absdiff/max(abs(emax), absdiff)

        # Update maximum
        if absdiff > self.maxabsdiff:
            self.maxabsdiff = absdiff
        if reldiff > self.maxreldiff:
            self.maxreldiff = reldiff
        if reldiff > CLOSE_ENOUGH:
            self.ndiffs += 1
            if self.verbose >= 2:
                print "%d %d" % index, edge, reldiff
            return False
        return True

    def diff(self, fhs):
        """Compare the given files."""
        if self.quick and not self.is_success():
            return

        nfiles = len(fhs)
        self.nfiles += nfiles

        # Stack of edges not yet diffed
        finished = set()
        queue = {}
        queuesize = 0

        # Iterate over the lines in each file (like paste(1))
        for lines in izip_longest(*fhs):
            self.nedges += 1
            # Parse line from each file
            for fno, line in enumerate(lines):
                if line is None:
                    continue
                # Parse line
                line = line.split()
                assert len(line) == 3
                index = (int(line[0], 10), int(line[1], 10))
                if self.symmetric and index[1] < index[0]:
                    index = (index[1], index[0])
                val = float(line[2])
                if isnan(val) or isinf(val):
                    self.ninvalid += 1
                    if self.quick:
                        return
                # Check queue
                if index in finished:
                    # We already found this edge in every file -- Collision!
                    self.ndups += 1
                    if self.quick:
                        return
                    else:
                        continue
                elif index not in queue:
                    # First time seeing this edge, create a new queue entry
                    queue[index] = [(val if i == fno else None)
                                    for i in xrange(nfiles)]
                    # Track queue size
                    queuesize += 1
                    if queuesize > self.maxqueue:
                        self.maxqueue = queuesize
                elif queue[index][fno] is not None:
                    # We already found this edge in this file -- Collision!
                    self.ndups += 1
                    if self.quick:
                        return
                    else:
                        continue
                else:
                    # Store value in existing queue entry
                    queue[index][fno] = val
                    # Check queue entry for completeness/correctness
                    result = self._diff_edge(index, queue[index])
                    if result is not None:
                        # Queue entry was complete, remove from queue
                        del queue[index]
                        queuesize -= 1
                        if self.quick and not result:
                            # We found an error while in quick mode
                            return
        # Check for differences in all remaining queue entries
        for index, entry in queue.iteritems():
            if entry is None:
                continue
            result = self._diff_edge(index, entry, incremental=False)
            if self.quick and not result:
                # We found an error while in quick mode
                return
        return

    def is_success(self):
        """Return True iff no error conditions have been detected."""
        return self.ninvalid == self.ndiffs == self.nmissing == self.ndups == 0

    def report(self):
        """Provide a report of the comparison, if verbosity allows. Returns
        is_success().

        """
        result = self.is_success()
        if result and self.verbose < 1:
            return result
        nedges = self.nedges if self.nedges else 1
        def pct(val):
            """Convert a counter to a percentage of edges."""
            return val*100./nedges

        print ("%d edges: %d differences (%.1f%%), %d missing (%.1f%%), " +
               "%d duplicates") % \
            (self.nedges, self.ndiffs, pct(self.ndiffs), self.nmissing,
             pct(self.nmissing), self.ndups)
        print "  %d implicit zeros (%.1f%%)" % \
            (self.nxzeros, pct(self.nxzeros))
        print "  %d files read" % (self.nfiles)
        print "  Maximum absolute difference = %g" % (self.maxabsdiff,)
        print "  Maximum relative difference = %g" % (self.maxreldiff,)
        print "  Maximum queue size = %d (%3.1f%%)" % \
            (self.maxqueue, pct(self.maxqueue))
        print "OK" if result else "Failed"

        return result

def main(argv):
    """Main entry point when run as a program."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Display differences between edge list files"
    )
    parser.add_argument('--asymmetric', dest='symmetric', action='store_false',
                        help="Do not assume a symmetric matrix")
    parser.add_argument('--quick', action='store_true',
                        help="Stop after first error")
    parser.add_argument('--quiet', '-q', action='count', default=0,
                        help="Show less output")
    parser.add_argument('--verbose', '-v', action='count', default=1,
                        help="Show more output")
    parser.add_argument('files', type=argparse.FileType('r'), nargs='*',
                        help="Files to compare", metavar='file')
    parser.set_defaults(
        files=[sys.stdin],
    )
    args = parser.parse_args(argv)
    args.verbose -= args.quiet
    differ = DiffEdgelists(symmetric=args.symmetric, verbose=args.verbose,
                           quick=args.quick)
    #import statprof
    #statprof.start()
    differ.diff(args.files)
    for filehandle in args.files:
        filehandle.close()
    #statprof.stop()
    #statprof.display()
    return 0 if differ.report() else 1

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
