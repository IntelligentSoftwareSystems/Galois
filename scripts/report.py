#!/usr/bin/env python
"""
Parse the output of an application into a csv file
Run a series of performance tests and pretty print and export csv
the results.

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

@author Donald Nguyen <ddn@cs.utexas.edu> 
"""

from __future__ import print_function
import sys
import os
import re
import optparse 
import collections

def main(options):
  frame = {
      'cols': set(),
      'rows': [],
      'row': collections.defaultdict(str)
    }

  def add_stat(f, key, value):
    try:
      if f['row'][key]:
        f['row'][key] = int(f['row'][key]) + int(value)
      else:
        f['row'][key] = value
    except ValueError:
      f['row'][key] = value
    f['cols'].add(key)
  def add_stat_l(f, key, value, loop):
    add_stat(f, key, value)
    add_stat(f, "%s-%s" % (key, loop), value)
  def do_start_line(f, m):
    if f['row']:
      f['rows'].append(f['row'])
      f['row'] = collections.defaultdict(str)
  def do_var_line(f, m):
    add_stat(f, m.group('key'), m.group('value'))
  def do_stat_line(f, m):
    add_stat_l(f, m.group('key'), m.group('value'), m.group('loop'))
  def do_dist_line(f, m):
    add_stat_l(f, m.group('key'), m.group('value'),
        '%s-%s' % (m.group('loop'), m.group('loopn')))

  table = {
      r'^RUN: Start': do_start_line,
      r'^RUN: Variable (?P<key>\S+) = (?P<value>\S+)': do_var_line,
      r'^STAT SINGLE (?P<key>\S+) (?P<loop>\S+) (?P<value>.*)': do_stat_line,
      r'^INFO: (?P<key>CommandLine) (?P<value>.*)': do_var_line,
      r'^STAT DISTRIBUTION (?P<loopn>\d+) (?P<key>\S+) (?P<loop>\S+) (?P<value>.*)': do_dist_line
      }
  
  matcher = [(re.compile(s), fn) for (s,fn) in table.iteritems()]
  for line in sys.stdin:
    for (regex, fn) in matcher:
      m = regex.match(line)
      if m:
        fn(frame, m)
        break
  if frame['row']:
    frame['rows'].append(frame['row'])
  
  if options.include:
    frame['cols'] = frame['cols'].intersect(options.include)
  elif options.exclude:
    frame['cols'] = frame['cols'].difference(options.exclude)
  frame['cols'] = sorted(frame['cols'])

  print(','.join(frame['cols']))
  for r in frame['rows']:
    print(','.join([str(r[c]) for c in frame['cols']]))


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
