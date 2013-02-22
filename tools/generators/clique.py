#!/usr/bin/env python
"""
Generates k-cliques

@section License

Copyright (C) 2013, The University of Texas at Austin. All rights reserved.
UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS SOFTWARE
AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY, FITNESS FOR ANY
PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF PERFORMANCE, AND ANY
WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF DEALING OR USAGE OF TRADE.
NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH RESPECT TO THE USE OF THE
SOFTWARE OR DOCUMENTATION. Under no circumstances shall University be liable
for incidental, special, indirect, direct or consequential damages or loss of
profits, interruption of business, or related expenses which may arise from use
of Software or Documentation, including but not limited to those resulting from
defects in Software and/or Documentation, or loss or inaccuracy of data of any
kind.

@author Donald Nguyen <ddn@cs.utexas.edu>
"""

from __future__ import print_function
import optparse
import random

def main(N, seed, options):
  random.seed(seed)
  max_weight = int(options.max_weight)

  for x in xrange(0, N):
    for y in xrange(0, N):
      if x == y:
        continue
      if max_weight:
        w = random.randint(1, max_weight)
        print('%d %d %d' % (x, y, w))
      else:
        print('%d %d' % (x, y))


if __name__ == '__main__':
  usage = 'usage: %prog [options] <N> [seed]'
  parser = optparse.OptionParser(usage=usage)
  parser.add_option('--max-weight', dest="max_weight",
      default=0, action='store',
      help='edge weights are uniformly selected from integers between (0, max-weight]. \
          If zero (default), generate unweighted graph.')
  (options, args) = parser.parse_args()
  if len(args) != 1:
    parser.error('missing arguments')
  if len(args) == 1:
    seed = 0
  else:
    seed = int(args[1])
  main(int(args[0]), seed, options)
