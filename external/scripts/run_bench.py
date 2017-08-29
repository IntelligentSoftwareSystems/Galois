#!/usr/bin/env python
#
# Normalize output of benchmark commands and passing of thread parameter

import optparse
import sys
import os
import socket
import subprocess
import signal

#signal.signal(signal.SIGQUIT, signal.SIG_IGN)

def backtick(s):
  proc = subprocess.Popen(s, stdout=subprocess.PIPE, shell=True)
  (out, err) = proc.communicate()
  return out

parser = optparse.OptionParser(usage='usage: %prog [options] -- <cmd>')
parser.add_option('-t', dest='threads', default=1, type='int')
parser.add_option('-r', dest='rounds', default=0, type='int')
parser.add_option('--dmp-chunksize', dest='dmpchunksize', default=1000, type='int')
(options, args) = parser.parse_args()

#nprocs = int(backtick("cat /proc/cpuinfo | grep processor | wc -l"))

os.environ['DMP_SCHEDULING_CHUNK_SIZE'] = str(options.dmpchunksize)
#os.environ['DMP_NUM_PHYSICAL_PROCESSORS'] = str(nprocs)
os.environ['DMP_NUM_PHYSICAL_PROCESSORS'] = str(options.threads)
os.environ['OMP_NUM_THREADS'] = str(options.threads)
os.environ['OMP_SCHEDULE'] = "dynamic,16"
os.environ['TBB_NUM_THREADS'] = str(options.threads)
os.environ['CILK_NWORKERS'] = str(options.threads)
os.environ['GALOIS_NUM_THREADS'] = str(options.threads)
os.environ['EXP_NUM_ROUNDS'] = str(options.rounds)

prog = args[0]
args = args[1:]
basename = os.path.basename(prog)

print("INFO: CommandLine %s %s" % (prog, ' '.join(args)))
print("INFO: Hostname %s" % socket.gethostname())
sys.stdout.flush()
os.execvp(prog, [prog] + args)
