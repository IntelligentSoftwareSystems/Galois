#!/usr/bin/env python
#
# Compact way to enumerate and filter and run benchmark suites.
# 
# Examples:
#  Run all pbbs benchmarks and inputs with 1 to 24 threads
#   run_suite.py -f pbbs -- --threads 24

import sys
import os
import signal
import optparse
import re
from collections import defaultdict


SpecPBBS = [
     {"prob": "breadthFirstSearch",
      "algos": ["deterministicBFS", "ndBFS", "g/nd0", "g/nd1", "g/pnd0", "g/pnd1", "g/d", "g/pd"], #, "serialBFS"],
#      "inputs": ["randLocalGraph_J_5_10000000", "rMatGraph_J_5_10000000", "3Dgrid_J_10000000", "USA-road-d.USA"],
      "inputs": ["randLocalGraph_J_5_10000000"],
      "extras":
        [
          {'algos': ['g/pnd0', 'g/nd0', 'g/d', 'g/pd'],
#            'arg': "Rounds::-r::1:10:1" 
            'arg': "Rounds::-r::1" 
          }
        ]
     },
     {"prob": "delaunayRefine",
      "algos": ["incrementalRefine", 'g/p', 'g/nd0', 'g/nd1', 'g/nd2', 'g/pnd0', 'g/pnd1', 'g/pnd2'],
      "inputs": [#"2DinCubeDelaunay_275000", "2DinCubeDelaunay_1000000", "r5M",
        "2DinCubeDelaunay_2500000"],
      "extras": 
        [
          {'algos': ["incrementalRefine", 'g/p', 'g/nd0', 'g/pnd0'],
#           'arg': "Rounds::-r::50,100:600:100"
           'arg': "Rounds::-r::500"
           }
        ]
      },
     {"prob": "delaunayTriangulation",
      "algos": ["incrementalDelaunay", "g/p", "g/nd0", "g/nd1", "g/pnd0", "g/pnd1"], #, "serialDelaunay"],
      "inputs": [#"2Dkuzmin_10M", 
        "2DinCube_10M", "2DinCube_10M-reordered.points"],
      "extras":
        [
          {'algos': ["incrementalDelaunay", 'g/p', 'g/nd0', 'g/pnd0'],
#           'arg': "Rounds::-r::50,100:600:100" 
           'arg': "Rounds::-r::100" 
          }
        ]
      },
     {"prob": "maximalIndependentSet",
      "algos": ["incrementalMIS", "ndMIS", "g/nd0", "g/nd1", "g/pnd0", "g/pnd1", "g/d", "g/pd"],# "serialMIS"],
#      "inputs": ["randLocalGraph_J_5_10000000", "rMatGraph_J_5_10000000", "3Dgrid_J_10000000", "USA-road-d.USA", "2d-2e26", "2d-2e26-ordered"],
      "inputs": ["randLocalGraph_J_5_10000000", "2d-2e26", "2d-2e26-ordered"],
      "extras":
        [
          {'algos': ["incrementalMIS", "g/nd0", "g/pnd0", "g/d", "g/pd"],
#           'arg': "Rounds::-r::1,25,50,100,200" 
           'arg': "Rounds::-r::25" 
          }
        ]
      }
#     ,
#     {"prob": "comparisonSort",
#      "algos": [#"quickSort",
#        "sampleSort", "serialSort"],
#      "inputs": ["randomSeq_100M_double", "exptSeq_100M_double", "almostSortedSeq_100M_double",
#                 "trigramSeq_100M", ("-p", "trigramSeq_100M")]},
#     {"prob": "convexHull",
#      "algos": ["quickHull", "serialHull"],
#      "inputs": ["2DinSphere_10M", "2Dkuzmin_10M", "2DonSphere_10M"]},
#     {"prob": "dictionary",
#      "algos": ["deterministicHash", "serialHash"],
#      "inputs": ["randomSeq_10M_int", "randomSeq_10M_100K_int", "exptSeq_10M_int", "trigramSeq_10M", "trigramSeq_10M_pair_int"]},
#     {"prob": "integerSort",
#      "algos": ["blockRadixSort", "serialRadixSort"],
#      "inputs": ["randomSeq_100M_int", "exptSeq_100M_int", "randomSeq_100M_int_pair_int", "randomSeq_100M_256_int_pair_int"]},
#     {"prob": "maximalMatching",
#      "algos": ["incrementalMatching", "ndMatching", "serialMatching"],
#      "inputs": ["randLocalGraph_E_5_10000000", "rMatGraph_E_5_10000000", "3Dgrid_E_10000000"]},
#     {"prob": "minSpanningTree",
#      "algos": ["parallelKruskal", "serialMST"],
#      "inputs": ["randLocalGraph_WE_5_10000000", "rMatGraph_WE_5_10000000", "2Dgrid_WE_10000000"]},
#     {"prob": "nBody",
#      "algos": ["parallelCK"],
#      "inputs": ["3DonSphere_1000000", "3DinCube_1000000", "3Dplummer_1000000"]},
#     {"prob": "nearestNeighbors",
#      "algos": ["octTree2Neighbors", "octTreeNeighbors"],
#      "inputs": [("-d 2 -k 1", "2DinCube_10M"), ("-d 2 -k 1", "2Dkuzmin_10M"), ("-d 3 -k 1", "3DinCube_10M"), ("-d 3 -k 1", "3DonSphere_10M"), ("-d 3 -k 10", "3DinCube_10M"), ("-d 3 -k 10", "3Dplummer_10M")]},
#     {"prob": "rayCast",
#      "algos": ["kdTree"],
#      "inputs": [("happyTriangles", "happyRays"), ("angelTriangles", "angelRays"), ("dragonTriangles", "dragonRays")]},
#     {"prob": "removeDuplicates",
#      "algos": ["deterministicHash", "serialHash"],
#      "inputs": [ "randomSeq_10M_int", "randomSeq_10M_100K_int", "exptSeq_10M_int", "trigramSeq_10M", "trigramSeq_10M_pair_int"]},
#     {"prob": "spanningTree",
#      "algos": ["incrementalST", "ndST", "serialST"],
#      "inputs": ["randLocalGraph_E_5_10000000", "rMatGraph_E_5_10000000", "2Dgrid_E_10000000"]},
#     {"prob": "suffixArray",
#      "algos": "serialKS",
#      "inputs": []}
    ]

def flatten(l):
  return ''.join(''.join(l).split())

def genPBBS(options):
  """Generate benches for PBBS suite."""
  def genextras(prob):
    """Generate table to correlate additional arguments with intended benchmarks"""
    extras = defaultdict(list)
    for entry in getfield(prob, 'extras', []):
      for key in entry['algos']:
        extras[key] += [entry['arg']]
    return extras

  benches = []
  baseinput = '%s/pbbs-0.1/inputs' % options.baseinput
  for prob in SpecPBBS:
    problem = prob["prob"]
    extras = genextras(prob)
    
    for algo in prob['algos']:
      salgo = algo.replace('/', '-')
      prog = "exp/suites/pbbs-0.1/apps/%s/%s/pbbs-%s-%s" % (problem, algo, problem, salgo)
      runargs = ['-x %s' % x for x in extras[algo]]
      for inp in prob['inputs']:
        args = getlist(inp)
        key = ["pbbs", problem, salgo, flatten(args)]
        b = {'key': '-'.join(key), 'prog': prog, 'wd': baseinput, 'args': args, 'runargs': runargs}
        benches.append(b)
  return benches


SpecParsec = [
    {'name': 'blackscholes', 'inputs': ["-1 in_64K.txt prices.txt", "-1 in_mid.txt prices.txt"]},#, '-1 in_10M.txt prices.txt']},
    {'name': 'bodytrack', 'inputs': ['sequenceB_261 4 261 4000 5 0 -1']},
    {'name': 'freqmine', 'inputs': ['kosarak_990k.dat 790']}
#      ['webdocs_250k.dat 11000', 'kosarak_250k.dat 220', 'kosarak_990k.dat 790']}
    ]

def genParsec(options):
  benches = []
  baseinput = '%s/parsec-2.1' % options.baseinput
  for prob in SpecParsec:
    name = prob['name']
    prog = 'exp/suites/parsec-2.1/pkgs/apps/%s/parsec-%s' % (name, name)
    wd = '%s/inputs/%s' % (baseinput, name)
    for inp in prob['inputs']:
      args = getlist(inp)
      key = ['parsec', name, flatten(args)]
      b = {'key': '-'.join(key), 'prog': prog, 'wd': wd, 'args': args, 'runargs': []}
      benches.append(b)
  return benches


def system(cmd, cwd=None, out=None):
  from threading import Thread
  class Printer(Thread):
    """Simple thread to take input and write it to stdout and file of our choice"""
    def __init__(self, inp, f):
      Thread.__init__(self)
      self.inp = inp
      self.f = f
    def run(self):
      while True:
        buf = self.inp.read(1)
        if not buf:
          break
      #for line in self.inp:
        sys.stdout.write(buf)
        f.write(buf)
        f.flush()
  
  from subprocess import PIPE, Popen, STDOUT
  if out:
    with open(out, 'w+') as f:
      process = Popen(cmd, stdout=PIPE, stderr=STDOUT, shell=True, cwd=cwd)
      printer = Printer(process.stdout, f)
      printer.start()
      process.wait()
      printer.join()
      status = process.returncode
  else:
    process = Popen(cmd, stderr=STDOUT, shell=True, cwd=cwd)
    process.wait()
    status = process.returncode

  if status < 0:
    sys.stderr.write("Child terminated by signal %d" % -status)
  return status


def getlist(l):
  """Turn a single string to singleton list"""
  if isinstance(l, str):
    return [l]
  else:
    return list(l)


def printbright(s):
  red = '\033[1;31m'
  endc = '\033[0m'
  print(red + s + endc)


def getfield(d, key, default):
  try:
    return d[key]
  except KeyError:
    return default


def runall(benches, runoptions, options):
  """
  Executes sequence of benches and keeps track of results.
  
  A bench is a dictionary with the following fields: 
    key     : unique identifier
    prog    : path to program to run
    wd      : working directory (default: .)
    args    : arguments to prog (default: [])
    runargs : arguments to run.py (default: [])
  """
  successes = filtered = skipped = failed = 0
  runpath = os.path.abspath(options.runpath)
  runbenchpath = os.path.abspath(options.runbenchpath)
  include = [re.compile(x) for x in options.include]
  exclude = [re.compile(x) for x in options.exclude]
  system('mkdir -p %s' % options.outdir)

  for bench in benches:
    logpath = '%s/%s' % (options.outdir, bench['key'])
    if os.path.exists(logpath):
      skipped += 1
    elif exclude and any([x.search(bench['key']) for x in exclude]):
      filtered += 1
    elif not include or all([x.search(bench['key']) for x in include]):
      runcmd = [runpath] + runoptions + getfield(bench, 'runargs', [])
      basecmd = [os.path.abspath(bench['prog'])] + getfield(bench, 'args', [])
      wd = getfield(bench, 'wd', '.')
      cmd = '%s -- %s -- %s' % (' '.join(runcmd), runbenchpath, ' '.join(basecmd))
      partial = '%s.partial' % logpath
      if options.verbose:
        printbright('(cd %s && %s) 2>&1 | tee -a %s' % (wd, cmd, partial))
       
      if system(cmd, cwd=wd, out=partial) == 0:
        system('mv %s %s' % (partial, logpath))
        successes += 1
      else:
        failed += 1
    else:
      filtered += 1

  total = successes + skipped + filtered + failed
  printbright('Total: %d (successes: %d filtered: %d skipped: %d failed: %d)' \
      % (total, successes, filtered, skipped, failed))


def main(args, options):
  benches = genPBBS(options) + genParsec(options)
  runall(benches, args, options)
  

if __name__ == '__main__':
  basepath = os.path.dirname(os.path.realpath(__file__))
  runpath = "%s/../../../scripts/run.py" % basepath
  runbenchpath = "%s/run_bench.py" % basepath

  signal.signal(signal.SIGQUIT, signal.SIG_IGN)
  # don't buffer stdout 
  sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

  parser = optparse.OptionParser(usage='usage: %prog [options] -- [run.py options]')
  parser.add_option('--baseinput', dest='baseinput', default='exp/suites',
      help='base directory for inputs')
  parser.add_option('-i', '--include', dest='include', default=[], action='append',
      help='include benchmarks whose keys match REGEX, repeated options form a conjunction', metavar='REGEX')
  parser.add_option('-e', '--exclude', dest='exclude', default=[], action='append',
      help='exclude benchmarks whose keys match REGEX, repeated options form a conjunction', metavar='REGEX')
  parser.add_option('--runpath', dest='runpath', default=runpath, help='path to run.py')
  parser.add_option('--runbenchpath', dest='runbenchpath', default=runbenchpath, help='path to run_bench.py')
  parser.add_option('--outdir', dest='outdir', default='plogs', help='directory to store output logs')
  parser.add_option('-v', '--verbose', dest='verbose', action='store_true', default=False)
  (options, args) = parser.parse_args()
  main(args, options)
