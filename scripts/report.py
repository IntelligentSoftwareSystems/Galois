#!/usr/bin/env python
"""
Parse the output of an application into a csv file

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
  class Row:
    def __init__(self):
      self.reset()
    def reset(self):
      self.r = collections.defaultdict(str)
      self.header = None
    def get(self, line, key):
      return line.split(',')[self.header.index(key)]


  cols = set()
  rows = []
  row = Row()

  def add_stat(key, value):
    # 'Threads' key duplicated so just directly assign instead of accumulate
    if key != "Threads":
      try:
        row.r[key] = int(row.r[key]) + int(value)
      except ValueError:
        row.r[key] = value
      except KeyError:
        row.r[key] = value
    else:
      row.r[key] = value
    cols.add(key)
  def add_stat_l(key, value, loop):
    add_stat(key, value)
    if loop != '(NULL)':
      add_stat("%s-%s" % (key, loop), value)
  def do_start_line(m):
    if row.r:
      rows.append(row.r)
      row.reset()
  def do_var_line(m):
    add_stat(m.group('key'), m.group('value'))
  def do_stat_header(m):
    row.header = m.group('value').split(',')
  def do_stat_line(m):
    v = m.group('value')
    add_stat_l(row.get(v, 'CATEGORY'), row.get(v, 'sum'), row.get(v, 'LOOP'))
  def do_old_stat_line(m):
    add_stat_l(m.group('key'), m.group('value'), m.group('loop'))
  def do_old_dist_line(m):
    add_stat_l(m.group('key'), m.group('value'),
        '%s-%s' % (m.group('loop'), m.group('loopn')))

  table = {
      r'^RUN: Start': do_start_line,
      r'^RUN: Variable (?P<key>\S+) = (?P<value>\S+)': do_var_line,
      r'^(RUN|INFO): (?P<key>CommandLine) (?P<value>.*)': do_var_line,
      r'^INFO: (?P<key>Hostname) (?P<value>.*)': do_var_line,
      r'^STAT SINGLE (?P<key>\S+) (?P<loop>\S+) (?P<value>.*)': do_old_stat_line,
      r'^STAT DISTRIBUTION (?P<loopn>\d+) (?P<key>\S+) (?P<loop>\S+) (?P<value>.*)': do_old_dist_line,
      r'^(?P<value>STATTYPE.*)': do_stat_header,
      r'^(?P<value>STAT,.*)': do_stat_line
      }
  
  matcher = [(re.compile(s), fn) for (s,fn) in table.iteritems()]
  for line in sys.stdin:
    for (regex, fn) in matcher:
      m = regex.match(line)
      if m:
        fn(m)
        break
  if row.r:
    rows.append(row.r)
  
  if options.include:
    cols = cols.intersect(options.include)
  elif options.exclude:
    cols = cols.difference(options.exclude)
  cols = sorted(cols)

  print(','.join(cols))
  for r in rows:
    print(','.join([str(r[c]) for c in cols]))


if __name__ == '__main__':
  parser = optparse.OptionParser(usage='usage: %prog [options]')
  parser.add_option('-i', '--include',
      dest="include", default=[], action='append',
      help='column to include in output. Multiple columns can be specified '
           + 'with multiple options or a comma separated list of columns.')
  parser.add_option('-e', '--exclude',
      dest="exclude", default=[], action='append',
      help='column to include in output. Multiple columns can be specified '
           + 'with multiple options or a comma separated list of columns.')

  (options, args) = parser.parse_args()
  if (not options.include and options.exclude) or (options.include and not options.exclude):
    parser.error('include and exclude are mutually exclusive')
  if args:
    parser.error('extra unknown arguments')

  main(options)
