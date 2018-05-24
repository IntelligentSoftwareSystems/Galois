#!/usr/bin/env python
"""
Generates random 2d points for use with Delaunay Triangulation.

"""

from __future__ import print_function
import sys
import optparse
import random

def main(npoints, seed, options):
  random.seed(seed)
  print('%d 2 0 0' % npoints)
  if not options.realistic:
    for i in xrange(npoints):
      x = random.random() * npoints
      y = random.random() * npoints
      # repr to get maximum precision in ASCII'ed representation
      print('%d %s %s 0' % (i, repr(x), repr(y)))
  else:
    # produce spatially coherent bins of about 10000 points each
    pass


if __name__ == '__main__':
  usage = 'usage: %prog [options] <num points> <seed>'
  parser = optparse.OptionParser(usage=usage)
  parser.add_option('-r', '--realistic', dest="realistic",
      default=False, action='store_true',
      help='use a more realistic strategy for generating points')
  (options, args) = parser.parse_args()
  if len(args) != 2:
    parser.error('missing arguments')
  if options.realistic:
    parser.error('not yet implemented')
  main(int(args[0]), int(args[1]), options)
