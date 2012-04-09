#!/usr/bin/env python
#

import sys
import os
import subprocess

D = [{"prob": "breadthFirstSearch",
      "algos": ["deterministicBFS", "ndBFS", "serialBFS"],
      "inputs": ["randLocalGraph_J_5_10000000", "rMatGraph_J_5_10000000", "3Dgrid_J_10000000"]},
#     {"prob": "comparisonSort",
#      "algos": [#"quickSort",
#        "sampleSort", "serialSort"],
#      "inputs": ["randomSeq_100M_double", "exptSeq_100M_double", "almostSortedSeq_100M_double",
#                 "trigramSeq_100M", ("-p", "trigramSeq_100M")]},
#     {"prob": "convexHull",
#      "algos": ["quickHull", "serialHull"],
#      "inputs": ["2DinSphere_10M", "2Dkuzmin_10M", "2DonSphere_10M"]},
     {"prob": "delaunayRefine",
      "algos": ["incrementalRefine", "ndIncrementalRefine"],
      "inputs": ["2DinCubeDelaunay_275000", "2DinCubeDelaunay_1000000", "2DinCubeDelaunay_2500000"]},
     {"prob": "delaunayTriangulation",
      "algos": ["ndIncrementalDelaunay", "incrementalDelaunay", "serialDelaunay"],
      "inputs": ["2DinCube_10M", "2Dkuzmin_10M"]},
#     {"prob": "dictionary",
#      "algos": ["deterministicHash", "serialHash"],
#      "inputs": ["randomSeq_10M_int", "randomSeq_10M_100K_int", "exptSeq_10M_int", "trigramSeq_10M", "trigramSeq_10M_pair_int"]},
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
#     {"prob": "nearestNeighbors",
#      "algos": ["octTree2Neighbors", "octTreeNeighbors"],
#      "inputs": [("-d 2 -k 1", "2DinCube_10M"), ("-d 2 -k 1", "2Dkuzmin_10M"), ("-d 3 -k 1", "3DinCube_10M"), ("-d 3 -k 1", "3DonSphere_10M"), ("-d 3 -k 10", "3DinCube_10M"), ("-d 3 -k 10", "3Dplummer_10M")]},
#     {"prob": "rayCast",
#      "algos": ["kdTree"],
#      "inputs": [("happyTriangles", "happyRays"), ("angelTriangles", "angelRays"), ("dragonTriangles", "dragonRays")]},
#     {"prob": "removeDuplicates",
#      "algos": ["deterministicHash", "serialHash"],
#      "inputs": [ "randomSeq_10M_int", "randomSeq_10M_100K_int", "exptSeq_10M_int", "trigramSeq_10M", "trigramSeq_10M_pair_int"]},
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
BASEDIR = os.path.dirname(os.path.realpath(__file__))

RUN_CMD = "%s/../../../scripts/run.py" % BASEDIR
RUN_PBBS_CMD = "%s/run_pbbs.sh" % BASEDIR

BASEINPUT = "exp/pbbs/inputs"
if "BASEINPUT" in os.environ:
  BASEINPUT = os.environ["BASEINPUT"]
NUMTHREADS = int(backtick("cat /proc/cpuinfo | grep processor | wc -l"))
if "NUMTHREADS" in os.environ:
  NUMTHREADS = int(os.environ["NUMTHREADS"])
BASENAME = os.path.basename(os.getcwd())


if not os.path.exists("Makefile"):
  sys.stderr.write("Execute this script from the base of your build directory\n")
  sys.exit(1)

system("mkdir -p " + RESULTDIR)

def run(key, cmd):
  global SKIPPED, FAILED
  log = "%s/%s" % (RESULTDIR, key)
  if os.path.exists(log) or "serial" in key or "delaunay" not in key:
    system("echo -n '\033[1;31m'")
    system("echo -n Skipping " + key)
    system("echo '\033[0m'")
    SKIPPED += 1
  else:
    top = NUMTHREADS + 1
    if "ndIncremental" in key:
      rounds = "-x Rounds::-r::1"
    else:
      rounds = "-x Rounds::-r::1,10,50,100:1000:100"
    ret = system("%s -t 1:%s %s -- %s %s | tee -a %s" % (RUN_CMD, top, rounds, RUN_PBBS_CMD, cmd, log))
    if ret != 0:
      system("rm %s" % log)
      FAILED += 1

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
          cmd += ["%s/%s" % (BASEINPUT, x)]
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
