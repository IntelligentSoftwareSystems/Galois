#!/usr/bin/env python
"""
Generates k-cliques

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
