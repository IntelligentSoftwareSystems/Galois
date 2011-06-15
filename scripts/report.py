#!/usr/bin/env python
#
# Parse the output of an application into a csv file
# Run a series of performance tests and pretty print and export csv
# the results.

from __future__ import print_function
import sys
import os
import re
import optparse 
import collections

def main(options):
  end_row = re.compile(r'^RUN: Start')
  run_var = re.compile(r'^RUN: Variable (\S+) = (\S+)')
  stat_var = re.compile(r'^STAT: (\S+) (.*)')

  cols = set()
  rows = []
  row = collections.defaultdict(str)
  for line in sys.stdin:
    m = run_var.match(line)
    if not m:
      m = stat_var.match(line)
    if m:
      row[m.group(1)] = m.group(2)
      cols.add(m.group(1))
    elif end_row.match(line) and row:
      rows.append(row)
      row = collections.defaultdict(str)
  if row:
    rows.append(row)
  
  if options.include:
    cols = cols.intersect(options.include)
  elif options.exclude:
    cols = cols.difference(options.exclude)
  cols = sorted(cols)

  print(','.join(cols))
  for r in rows:
    print(','.join([r[c] for c in cols]))


if __name__ == '__main__':
  parser = optparse.OptionParser(usage='usage: %prog [options]')
  parser.add_option('-i', '--include', dest="include", default=[], action='append',
      help='column to include in output. Multiple columns can be specified with multiple options'
           + 'or a comma separated list of columns.')
  parser.add_option('-e', '--exclude', dest="exclude", default=[], action='append',
      help='column to include in output. Multiple columns can be specified with multiple options'
           + 'or a comma separated list of columns.')
  (options, args) = parser.parse_args()
  if (not options.include and options.exclude) or (options.include and not options.exclude):
    parser.error('include and exclude are mutually exclusive')
  if args:
    parser.error('extra unknown arguments')

  main(options)
