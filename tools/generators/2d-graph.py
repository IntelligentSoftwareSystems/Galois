#!/usr/bin/env python
"""
Generates 2D graphs 

"""

from __future__ import print_function
import sys
import optparse
import random
import collections

class Random:
  def __init__(self, N, max_weight):
    self.N = N
    self.max_weight = max_weight
    self.indices = range(0, N*N)
    random.shuffle(self.indices)
  def addEdge(self, ax, ay, bx, by):
    da = self.xy2d(ax, ay)
    db = self.xy2d(bx, by)
    w = random.randint(1, self.max_weight)
    print('a %d %d %d' % (da+1, db+1, w))
  def xy2d(self, x, y):
    return self.indices[y * self.N + x]


# From Wikipedia: http://en.wikipedia.org/wiki/Hilbert_curve (7/9/12)
def xy2d(n, x, y):
  rx = 0
  ry = 0
  d = 0
  s = n / 2
  while s > 0:
    rx = (x & s) > 0
    ry = (y & s) > 0
    d += s * s * ((3 * rx) ^ ry)
    (x, y) = rot(s, x, y, rx, ry)
    s /= 2
  return d

def d2xy(n, d):
  t = d
  s = 1
  x = 0
  y = 0
  while s < n:
    rx = 1 & (t/2)
    ry = 1 & (t ^ rx)
    (x, y) = rot(s, x, y, rx, ry)
    x += s * rx
    y += s * ry
    t /= 4
    s *= 2
  return (x,y)

def rot(n, x, y, rx, ry):
  if ry == 0:
    if rx == 1:
      x = n-1 - x
      y = n-1 - y
    t = x
    x = y
    y = t
  return (x,y)

class Hilbert:
  def __init__(self, N, max_weight):
    self.N = N
    self.max_weight = max_weight
  def addEdge(self, ax, ay, bx, by):
    da = xy2d(self.N, ax, ay)
    db = xy2d(self.N, bx, by)
    w = random.randint(1, self.max_weight)
    print('a %d %d %d' % (da+1, db+1, w))


def main(N, seed, options):
  random.seed(seed)
  max_weight = int(options.max_weight)
  num_nodes = N * N 
  num_edges = num_nodes * 2 - 2 * N

  print('p sp %d %d' % (num_nodes, num_edges))

  if options.reorder:
    G = Hilbert(N, max_weight)
  else:
    G = Random(N, max_weight)

  for x in xrange(0, N):
    for y in xrange(0, N):
      if x != N - 1:
        G.addEdge(x, y, x+1, y)
      if y != N - 1:
        G.addEdge(x, y, x, y+1)


if __name__ == '__main__':
  usage = 'usage: %prog [options] <N> <seed>'
  parser = optparse.OptionParser(usage=usage)
  parser.add_option('--max-weight', dest="max_weight",
      default=10000, action='store',
      help='edge weights are uniformly selected from integers \
          between (0, max-weight]')
  parser.add_option('--reorder', dest="reorder",
      default=False, action="store_true",
      help="reorder nodes with space filling curve")
  (options, args) = parser.parse_args()
  if len(args) != 2:
    parser.error('missing arguments')
  main(int(args[0]), int(args[1]), options)
