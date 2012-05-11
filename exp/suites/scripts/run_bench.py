#!/usr/bin/env python
#
# Normalize output of benchmark commands and passing of thread parameter

import optparse
import sys
import os
import socket
import subprocess

def backtick(s):
  proc = subprocess.Popen(s, stdout=subprocess.PIPE, shell=True)
  (out, err) = proc.communicate()
  return out

parser = optparse.OptionParser(usage='usage: %prog [options] -- <cmd>')
parser.add_option('-t', dest='threads', default=1, type='int')
parser.add_option('-r', dest='rounds', default=1, type='int')
(options, args) = parser.parse_args()

nprocs = int(backtick("cat /proc/cpuinfo | grep processor | wc -l"))

#export DMP_SCHEDULING_CHUNK_SIZE=1000
os.environ['DMP_NUM_PHYSICAL_PROCESSORS'] = str(nprocs)
os.environ['OMP_NUM_THREADS'] = str(options.threads)
os.environ['OMP_SCHEDULE'] = "dynamic,16"
os.environ['TBB_NUM_THREADS'] = str(options.threads)
os.environ['CILK_NWORKERS'] = str(options.threads)
os.environ['GALOIS_NUM_THREADS'] = str(options.threads)
os.environ['EXP_NUM_ROUNDS'] = str(options.rounds)

prog = args[0]
args = args[1:]
basename = os.path.basename(prog)

if basename == 'parsec-blackscholes':
  args = [str(options.threads)] + args
elif basename == 'parsec-bodytrack':
  args = args + [str(options.threads)]

print("INFO: CommandLine %s %s" % (prog, ' '.join(args)))
print("INFO: Hostname %s" % socket.gethostname())
sys.exit(subprocess.call([prog] + args))
