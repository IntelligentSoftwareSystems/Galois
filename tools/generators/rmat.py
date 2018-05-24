#!/usr/bin/env python
"""
Wrapper around GTgraph program.

"""

from __future__ import print_function
import optparse
import random
import tempfile
import subprocess;

def writeConfig(f, options):
  f.write('n %s\n' % options.num_nodes)
  f.write('m %s\n' % options.num_edges)
  f.write('a %s\n' % options.a)
  f.write('b %s\n' % options.b)
  f.write('c %s\n' % options.c)
  f.write('d %s\n' % options.d)
  f.write('MAX_WEIGHT %s\n' % options.max_weight)
  f.write('MIN_WEIGHT %s\n' % options.min_weight)
  f.write('SELF_LOOPS 0\n')
  f.write('STORE_IN_MEMORY 0\n')
  f.write('SORT_EDGELISTS 0\n')
  f.write('SORT_TYPE 1\n')
  f.write('WRITE_TO_FILE 1\n')
  f.flush()


def main(GTgraph, output, options):
  with tempfile.NamedTemporaryFile(delete=False) as f:
    writeConfig(f, options)
    subprocess.check_call([GTgraph, '-c', f.name, '-o', output])


if __name__ == '__main__':
  usage = 'usage: %prog [options] <GTgraph binary> <output>'
  parser = optparse.OptionParser(usage=usage)
  parser.add_option('-n', '--num-nodes', dest="num_nodes", action='store', help='Number of nodes.')
  parser.add_option('-m', '--num-edges', dest="num_edges", action='store', help='Number of edges.')
  parser.add_option('-a', '--param-a', dest="a", action='store', default='0.45', help='Parameter a of rmat.')
  parser.add_option('-b', '--param-b', dest="b", action='store', default='0.15', help='Parameter b of rmat.')
  parser.add_option('-c', '--param-c', dest="c", action='store', default='0.15', help='Parameter c of rmat.')
  parser.add_option('-d', '--param-d', dest="d", action='store', default='0.25', help='Parameter d of rmat.')
  parser.add_option('--max-weight', dest="max_weight",
      default=100, action='store',
      help='edge weights are uniformly selected from integers between [min-weight, max-weight].')
  parser.add_option('--min-weight', dest="min_weight",
      default=0, action='store',
      help='edge weights are uniformly selected from integers between [min-weight, max-weight].')

  (options, args) = parser.parse_args()
  if not options.num_nodes:
    parser.error('missing num nodes')
  if not options.num_edges:
    parser.error('missing num edges')
  if len(args) != 2:
    parser.error('missing required arguments')
  main(args[0], args[1], options)
