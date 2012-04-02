#!/usr/bin/env python
#

import sys
import os
import subprocess

D = [{"prob": "breadthFirstSearch",
      "algos": ["deterministicBFS", "ndBFS", "serialBFS"],
      "inputs": ["randLocalGraph_J_5_10000000", "rMatGraph_J_5_10000000", "3Dgrid_J_10000000"]},
     {"prob": "comparisonSort",
      "algos": [#"quickSort",
        "sampleSort", "serialSort"],
      "inputs": ["randomSeq_100M_double", "exptSeq_100M_double", "almostSortedSeq_100M_double",
                 "trigramSeq_100M", ("-p", "trigramSeq_100M")]},
     {"prob": "convexHull",
      "algos": ["quickHull", "serialHull"],
      "inputs": ["2DinSphere_10M", "2Dkuzmin_10M", "2DonSphere_10M"]},
     {"prob": "delaunayRefine",
      "algos": ["incrementalRefine"],
      "inputs": ["2DinCubeDelaunay_275000", "2DinCubeDelaunay_1000000", "2DinCubeDelaunay_2500000"]},
     {"prob": "delaunayTriangulation",
      "algos": ["incrementalDelaunay", "serialDelaunay"],
      "inputs": ["2DinCube_10M", "2Dkuzmin_10M"]},
     {"prob": "dictionary",
      "algos": ["deterministicHash", "serialHash"],
      "inputs": ["randomSeq_10M_int", "randomSeq_10M_100K_int", "exptSeq_10M_int", "trigramSeq_10M", "trigramSeq_10M_pair_int"]},
#     {"prob": "integerSort",
#      "algos": ["blockRadixSort", "serialRadixSort"],
#      "inputs": ["randomSeq_100M_int", "exptSeq_100M_int", "randomSeq_100M_int_pair_int", "randomSeq_100M_256_int_pair_int"]},
     {"prob": "maximalIndependentSet",
      "algos": ["incrementalMIS", "ndMIS", "serialMIS"],
      "inputs": ["randLocalGraph_J_5_10000000", "rMatGraph_J_5_10000000", "3Dgrid_J_10000000"]},
     {"prob": "maximalMatching",
      "algos": ["incrementalMatching", "ndMatching", "serialMatching"],
      "inputs": ["randLocalGraph_E_5_10000000", "rMatGraph_E_5_10000000", "3Dgrid_E_10000000"]},
#     {"prob": "minSpanningTree",
#      "algos": ["parallelKruskal", "serialMST"],
#      "inputs": ["randLocalGraph_WE_5_10000000", "rMatGraph_WE_5_10000000", "2Dgrid_WE_10000000"]},
#     {"prob": "nBody",
#      "algos": ["parallelCK"],
#      "inputs": ["3DonSphere_1000000", "3DinCube_1000000", "3Dplummer_1000000"]},
     {"prob": "nearestNeighbors",
      "algos": ["octTree2Neighbors", "octTreeNeighbors"],
      "inputs": [("-d 2 -k 1", "2DinCube_10M"), ("-d 2 -k 1", "2Dkuzmin_10M"), ("-d 3 -k 1", "3DinCube_10M"), ("-d 3 -k 1", "3DonSphere_10M"), ("-d 3 -k 10", "3DinCube_10M"), ("-d 3 -k 10", "3Dplummer_10M")]},
     {"prob": "rayCast",
      "algos": ["kdTree"],
      "inputs": [("happyTriangles", "happyRays"), ("angelTriangles", "angelRays"), ("dragonTriangles", "dragonRays")]},
     {"prob": "removeDuplicates",
      "algos": ["deterministicHash", "serialHash"],
      "inputs": [ "randomSeq_10M_int", "randomSeq_10M_100K_int", "exptSeq_10M_int", "trigramSeq_10M", "trigramSeq_10M_pair_int"]},
     {"prob": "spanningTree",
      "algos": ["incrementalST", "ndST", "serialST"],
      "inputs": ["randLocalGraph_E_5_10000000", "rMatGraph_E_5_10000000", "2Dgrid_E_10000000"]},
     {"prob": "suffixArray",
      "algos": "serialKS",
      "inputs": []}
    ]

def backtick(s):
  proc = subprocess.Popen(s, stdout=subprocess.PIPE, shell=True)
  (out, err) = proc.communicate()
  return out

def system(s):
  status = subprocess.call(s, shell=True)
  if status < 0:
    sys.stderr.write("Child terminated by signal %d" % -status)
  return status

SKIPPED = 0
FAILED = 0
RESULTDIR = "plogs"

BASEINPUTS = "exp/pbbs/inputs"
if "BASEINPUT" in os.environ:
  BASEINPUTS = os.environ["BASEINPUT"]
if "NUMTHREADS" in os.environ:
  NUMTHREADS = int(os.environ["NUMTHREADS"])
else:
  NUMTHREADS = int(backtick("cat /proc/cpuinfo | grep processor | wc -l"))
BASENAME = os.path.basename(os.getcwd())


if not os.path.exists("Makefile"):
  sys.stderr.write("Execute this script from the base of your build directory\n")
  sys.exit(1)

system("mkdir -p " + RESULTDIR)

def run(key, cmd):
  global SKIPPED, FAILED
  log = "%s/%s" % (RESULTDIR, key)
  if os.path.exists(log) or "serial" in key:
    system("echo -n '\033[1;31m'")
    system("echo -n Skipping " + key)
    system("echo '\033[0m'")
    SKIPPED += 1
  else:
    system("echo -n '\033[1;31m'")
    system("echo -n \"%s\"" % cmd)
    system("echo '\033[0m'")
    top = NUMTHREADS + 1
    if "serial" in key:
      top = 2
    for t in range(1, top):
      if os.path.exists("/tmp/stopme"):
        sys.exit(0)
      system("echo 'RUN: Start' | tee -a %s" % log)
      system("echo 'RUN: Variable Threads = %d' | tee -a %s" % (t, log))
      system("echo 'INFO: CommandLine %s' | tee -a %s" % (cmd, log))
      system("echo 'INFO: Hostname %s' | tee -a %s" % (os.uname()[1], log))
      system("echo 'STAT SINGLE Kind (null) %s' | tee -a %s" % (BASENAME, log))
      os.environ["OMP_NUM_THREADS"] = str(t)
      os.environ["OMP_SCHEDULE"] = "dynamic,16"
      os.environ["TBB_NUM_THREADS"] = str(t)
      os.environ["CILK_NWORKERS"] = str(t)
      os.environ["GALOIS_NUM_THREADS"] = str(t)
      ret = system("set -e; %s 2>&1 | tee -a %s" % (cmd, log))
      if ret != 0:
        system("rm %s" % log)
        FAILED += 1
        break

# Simple function to turn single string to singleton list
def getlist(l):
  if isinstance(l, str):
    return [l]
  else:
    return l


for prob in D:
  for algo in prob["algos"]:
    for inp in getlist(prob["inputs"]):
      p = prob["prob"]
      prog = "exp/pbbs/apps/%s/%s/pbbs-%s-%s" % (p, algo, p, algo)
      cmd = [prog]
      key = [p, algo]
      for x in getlist(inp):
        if x.startswith("-"):
          cmd += [x]
          key += ["".join(x.split())]
        else:
          cmd += ["%s/%s" % (BASEINPUTS, x)]
          key += [x]
      run("-".join(key), " ".join(cmd))

if FAILED > 0:
  system("echo -n '\033[1;32m'")
  system("echo -n \"SOME FAILED (skipped %d) (failed %d)\"" % (SKIPPED, FAILED))
  system("echo '\033[0m'")
elif SKIPPED > 0:
  system("echo -n '\033[1;32m'")
  system("echo -n \"SOME OK (skipped %d)\"" % SKIPPED)
  system("echo '\033[0m'")
else:
  system("echo -n '\033[1;32m'")
  system("echo -n ALL OK")
  system("echo '\033[0m'")
