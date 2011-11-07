#!/usr/bin/env python
#
# Run an application multiple times, varying parameters like
# number of threads, etc

from __future__ import print_function
import sys
import os
import subprocess
import optparse
import shlex
import signal


def die(s):
  sys.stderr.write(s)
  sys.exit(1)


def print_bright(s):
  red = '\033[1;31m'
  endc = '\033[0m'
  print(red + s + endc)


# Parses thread range from G.options
# Grammar:
#  R := R,R
#     | N
#     | N:N
#     | N:N:N
#  N := an integer
def get_range(s):
  # Parsing strategy: greedily parse integers with one character
  # lookahead to figure out exact category
  s = s + ' ' # append special end marker
  retval = []
  curnum = -1
  curseq = []
  for i in range(len(s)):
    if s[i].isdigit() and curnum < 0:
      curnum = i
    elif s[i].isdigit():
      pass
    elif s[i] == ',' or s[i] == ' ':
      if curnum < 0:
        break
      num = int(s[curnum:i])
      if len(curseq) == 0:
        retval.append(num)
      elif len(curseq) == 1:
        retval.extend(range(curseq[0], num + 1))
      elif len(curseq) == 2:
        retval.extend(range(curseq[0], curseq[1] + 1, num))
      else:
        break
      curnum = -1
      curseq = []
    elif s[i] == ':' and curnum >= 0:
      curseq.append(int(s[curnum:i]))
      curnum = -1
    else:
      break
  else:
    return sorted(set(retval))
  die('error parsing range: %s\n' % s)


# Like itertools.product but for one iterable of iterables
# rather than an argument list of iterables
def product(args):
  pools = map(tuple, args)
  result = [[]]
  for pool in pools:
    result = [x+[y] for x in result for y in pool]
  for prod in result:
    yield tuple(prod)


def main(args, options):
  variables = []
  ranges = []
  for extra in options.extra:
    (name, arg, r) = extra.split('::')
    variables.append((name, arg))
    ranges.append(get_range(r))

  for prod in product(ranges):
    cmd = [args[0]]
    values = []
    for ((name, arg), value) in zip(variables, prod):
      cmd.extend([arg, str(value)])
      values.append((name, str(value)))
    cmd.extend(args[1:])

    for run in range(options.runs):
      print('RUN: Start')
      print_bright('RUN: Executing %s' % ' '.join(cmd))
      for (name, value) in values:
        print('RUN: Variable %s = %s' % (name, value))

      import subprocess, datetime, os, time, signal
      sys.stdout.flush()
      if options.timeout:
        start = datetime.datetime.now()
        process = subprocess.Popen(cmd)
        while process.poll() is None:
          time.sleep(5)
          now = datetime.datetime.now()
          diff = (now-start).seconds
          if diff > options.timeout:
            os.kill(process.pid, signal.SIGKILL)
            os.waitpid(-1, os.WNOHANG)
            print("RUN: Variable Timeout = %d\n" % (diff*1000))
            break
        retcode = process.returncode
      else:
        retcode = subprocess.call(cmd)
      if retcode != 0:
        print("INFO: CommandLine %s\n", % ' ',join(cmd))
        print("RUN: Error command: %s\n" % cmd)
        #sys.exit(1)


if __name__ == '__main__':
  signal.signal(signal.SIGQUIT, signal.SIG_IGN)
  parser = optparse.OptionParser(usage='usage: %prog [options] <command line> ...')
  parser.add_option('-t', '--threads', dest="threads", default="1",
      help='range of threads to use. A range is R := R,R | N | N:N | N:N:N where N is an integer.')
  parser.add_option('-r', '--runs', default=1, type="int",
      help="set number of runs")
  parser.add_option('-x', '--extra', dest="extra", default=[], action='append',
      help='add another parameter to range over (format: <name>::<arg>::<range>). E.g., delta::-delta::1,5')
  parser.add_option('-o', '--timeout', dest="timeout", default=0, type='int',
      help="timeout a run after SEC seconds", metavar='SEC')
  (options, args) = parser.parse_args()
  if not args:
    parser.error('need command to run')
  options.extra.insert(0, '%s::%s::%s' % ('Threads', '-t', options.threads))
  main(args, options)
