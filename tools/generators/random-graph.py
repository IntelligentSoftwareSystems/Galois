#!/usr/bin/env python
"""
Generates random graphs

@section License

Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
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

@author Andrew Lenharth <andrewl@lenharth.org>
"""

from __future__ import print_function
import sys
import optparse
import random

#my $num = 268435456; #28
#my $num = 134217728; #27
#my $num =  67108864; #26
#my $num =  33554432; #25
#my $num =  16777216; #24
#my $num =   8388608; #23
#my $num =   4194304; #22
#my $num =   2097152; #21

def main(num_nodes, seed, options):
  random.seed(seed)
  max_weight = int(options.max_weight)
  num_edges = int(options.density) * num_nodes

  print('p sp %d %d' % (num_nodes, num_edges))

  def nextN():
    return random.randint(0, num_nodes - 1)

  def nextE():
    return random.randint(1, max_weight)

  # Form a connected graph
  for index in xrange(num_nodes):
    print('a %d %d %d' % (index, index + 1, nextE()))
  print('a %d %d %d' % (num_nodes - 1, 0, nextE()))

  for index in xrange(num_edges - num_nodes):
    print('a %d %d %d' % (nextN(), nextN(), nextE()))


if __name__ == '__main__':
  usage = 'usage: %prog [options] <num nodes> <seed>'
  parser = optparse.OptionParser(usage=usage)
  parser.add_option('--max-weight', dest="max_weight",
      default=10000, action='store',
      help='edge weights are uniformly selected from integers \
          between (0, max-weight]')
  parser.add_option('--density', dest="density",
      default=4, action='store',
      help='total number of edges is numnodes * density')
  (options, args) = parser.parse_args()
  if len(args) != 2:
    parser.error('missing arguments')
  main(int(args[0]), int(args[1]), options)
