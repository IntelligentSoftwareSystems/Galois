#!/usr/bin/env python
"""
Like cat but optionally add key-values after 'RUN: Start'. Useful with report.py.

@section License

Copyright (C) 2012, The University of Texas at Austin. All rights reserved.
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
import sys
import os
import re
import optparse 
import collections

def process(fh, options):
  regex = re.compile(r'^RUN: Start')
  pairs = [kv.split('=') for kv in options.add]
  text = '\n'.join(['RUN: Variable %s = %s' % (k, v) for (k,v) in pairs])

  for line in fh:
    print(line, end='')
    if regex.match(line):
      print(text)


if __name__ == '__main__':
  parser = optparse.OptionParser(usage='usage: %prog [options]')
  parser.add_option('-a', '--add-column',
      dest="add", default=[], action='append',
      help='column to include in output. Multiple columns can be specified '
           + 'with multiple options or a comma separated list of columns. '
           + 'Example: --add-column Version=Good')

  (options, args) = parser.parse_args()

  if args:
    for f in args:
      with open(f) as fh:
        process(fh, options)
  else:
    process(sys.stdin, options)
