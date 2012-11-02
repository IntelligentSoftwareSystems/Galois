#!/usr/bin/env python
from __future__ import print_function
import sys
import re

KeepAll = True

ToKeepV0 = ['Function', 'Module',
  'CPU_CLK_UNHALTED.REF', 'INST_RETIRED.ANY', 'MEM_LOAD_RETIRED.LLC_MISS']
ToKeepV1 = ['Function', 'Module',
  'CPU_CLK_UNHALTED.REF_P', 'INST_RETIRED.ANY', 'OFFCORE_RESPONSE_0.DEMAND_DATA_RD.ANY_LLC_MISS']
Columns = []
Header = []
Version = 0

def decode(func, module):
  if func.startswith('GaloisRuntime:'):
    return 'runtime'
  if module.find('libcilkrts.so') != -1:
    return 'runtime'
  if module == 'vmlinux':
    return 'system'
  if module.startswith('libc'):
    return 'runtime'
  if module.endswith('.so'):
    return 'runtime'
#  print('UNKNOWN %s %s' % (func, module))
#  if module == 'delaunayrefinement':
#    return 'user'
  return 'user'

def parse(line, first):
  global Columns, Version, Header
  
  if first:
    Header = re.sub(r':\w+ Event Count', '', line).split()
    if Header:
      try:
        Columns = [Header.index(x) for x in ToKeepV0]
        Version = 0
      except ValueError:
        Columns = [Header.index(x) for x in ToKeepV1]
        Version = 1
      return False
    else:
      return True
  if Version == 0:
    values = line.split('\t')
  else:
    if line.startswith('-'):
      return False
    values = line.split()
    d = len(values) - len(Header)
    values = [' '.join(values[0:d+1])] + values[d+1:]
  if KeepAll:
    for (h,v) in zip(Header, values):
      if h == "Function" or h == "Module":
        continue
      print('RUN: Variable PC_%s = %s' % (h, v))
  else:
    (func, module, cycles, insts, llcm) = [values[x] for x in Columns]
    kind = decode(func, module)
    print('RUN: Variable %sCycles = %s' % (kind, cycles))
    print('RUN: Variable %sInsts = %s' % (kind, insts))
    print('RUN: Variable %sLlcm = %s' % (kind, llcm))
  return False

def main():
  indata = False
  first = False
  for line in sys.stdin:
    if line.startswith('THREADS'):
      if indata:
        indata = False
        first = False
      else:
        indata = True
        first = True
      continue
    if line.startswith('RUN: CUT START'):
      indata = True
      first = True
      continue
    if line.startswith('RUN: '):
      indata = False
      first = False
      if line.startswith('RUN: CUT STOP'):
        continue

    if indata:
      first = parse(line, first)
    else:
      print(line, end='')

if __name__ == '__main__':
  main()
