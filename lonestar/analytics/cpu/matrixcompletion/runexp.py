#!/usr/bin/env python

import os
import sys
import subprocess
import math

RootDir = '/net/gilbert/workspace/ddn/build'
RunPy = '%s/w/GaloisDefault/scripts/run.py' % os.environ['HOME']
CommandTable = [{
  'name': 'bgg',
  'lambda': '0.01',
  'learningRate': '0.0005',
  'decayRate': '0.001',
  'time': 500,
  'secondsPerIteration': {'nomad': 1, 'galois': 0.25, 'graphlab': 0.5 },
  'nomadSecondsPerIteration': 10,
  'galoisOption': '-itemsPerBlock 200 -usersPerBlock 1024',
}, {
  'name': 'netflix',
  'lambda': '0.05',
  'learningRate': '0.012',
  'decayRate': '0.015',
  'time': 1000,
  'secondsPerIteration': {'nomad': 10, 'galois': 1, 'graphlab': 30 },
  'galoisOption': '-itemsPerBlock 150 -usersPerBlock 2048'
}, {
  'name': 'yahoo',
  'lambda': '1.0',
  'learningRate': '0.00075',
  'decayRate': '0.01',
  'time': 1000,
  'secondsPerIteration': {'nomad': 25, 'galois': 25, 'graphlab': 100 },
  #'galoisOption': '-itemsPerBlock 1500 -usersPerBlock 40000' # transpose
  'galoisOption': '-itemsPerBlock 4625 -usersPerBlock 7025'
}]

def galoisCommand(c):
  # exp/apps/sgd/sgd-ddn /net/faraday/workspace/inputs/weighted/bipartite/bgg.gr -t 20 -algo blockedEdge -lambda 0.01 -learningRate 0.0005 -decayRate 0.001 -learningRateFunction Purdue -itemsPerBlock 200 -usersPerBlock 1024 -fixedRounds 400 -useExactError
  cmd = os.path.join(RootDir, 'default/exp/apps/matrixcompletion/mc-ddn')
  input = os.path.join('/net/faraday/workspace/inputs/weighted/bipartite', c['name'] + '.gr')
  
  iterations = int(math.ceil(c['time'] / c['secondsPerIteration']['galois']))
  opts = ['-useSameLatentVector', '-algo blockedEdge', '-useExactError', '-fixedRounds', str(iterations)]
  opts += ['-learningRateFunction purdue']
  opts += [c['galoisOption']]
  nparams = ['-lambda', '-learningRate', '-decayRate']
  cparams = ['lambda', 'learningRate', 'decayRate']
  params = [(n, c[p]) for (n,p) in zip(nparams, cparams)]

  s = [RunPy, '-t 20,40', '--', cmd, input]
  #s = [RunPy, '-t 20', '--', cmd, input]
  s += [v for s in params for v in s]
  s += opts
  subprocess.call(' '.join(s), shell=True)

def graphlabCommand(c):
  # toolkits/collaborative_filtering/sgd --matrix /net/faraday/workspace/ddn/inputs/graphlab/bgg --D 100 --lambda 0.01 --gamma 0.0005 --step_dec 0.001 --ncpus 40 --max_iter 400
  cmd = os.path.join(RootDir, 'graphlab/toolkits/collaborative_filtering/sgd')
  input = os.path.join('/net/faraday/workspace/ddn/inputs/graphlab', c['name'])
  
  iterations = int(math.ceil(c['time'] / c['secondsPerIteration']['graphlab']))
  opts = ['--D 100', '--max_iter', str(iterations)]
  nparams = ['--lambda', '--gamma', '--step_dec']
  cparams = ['lambda', 'learningRate', 'decayRate']
  params = [(n, c[p]) for (n,p) in zip(nparams, cparams)]

  s = [RunPy, '--no-default-thread', '-x Threads::--ncpus::20,40', '--', cmd, input]
  s += [v for s in params for v in s]
  s += opts
  subprocess.call(' '.join(s), shell=True)

def nomadCommand(c):
  # mpirun ./nomad_double --path /net/faraday/workspace/ddn/inputs/nomad/bgg --nthreads 40 --lrate 0.0005  --drate 0.001  --dim  100  --reg 0.01 --timeout 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200
  cmd = os.path.join(RootDir, 'nomad/nomad_double')
  input = os.path.join('/net/faraday/workspace/ddn/inputs/nomad', c['name'])
  iterations = int(math.ceil(c['time'] / c['secondsPerIteration']['nomad']))
  timeouts = [str(c['secondsPerIteration']['nomad'] * x) for x in range(1, iterations + 1)]
  opts = ['--dim 100', '--timeout', ' '.join(timeouts)]
  nparams = ['--reg', '--lrate', '--drate']
  cparams = ['lambda', 'learningRate', 'decayRate']
  params = [(n, c[p]) for (n,p) in zip(nparams, cparams)]

  s = [RunPy, '--no-default-thread', '--append-arguments', '-x Threads::--nthreads::20,40', '--', 'mpirun', cmd, '--path', input]
  s += [v for s in params for v in s]
  s += opts
  subprocess.call(' '.join(s), shell=True)

def main():
  sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
  for c in CommandTable:
    #for t in [galoisCommand, graphlabCommand, nomadCommand]:
    for t in [galoisCommand]:
      t(c)


if __name__ == '__main__':
  main()
