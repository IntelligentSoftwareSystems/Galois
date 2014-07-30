#!/usr/bin/env python

import subprocess

RunPy = '~/w/GaloisDefault/scripts/run.py'

# on peltier
# filename                                     size    bs-time   ikdg-time
InputRaw = """
\'\'                                              100   6.774607s  15.407207s
/net/faraday/workspace/inputs/fl/audikw_1.gr   9437  20.188667s  13.846929s
/net/faraday/workspace/inputs/fl/inline_1.gr   5038  12.303275s   8.928665s
/net/faraday/workspace/inputs/fl/Emilia_923.gr 9232  17.784135s   6.517044s
/net/faraday/workspace/inputs/fl/boneS10.gr    9149  14.014836s   3.225513s
/net/faraday/workspace/inputs/fl/ldoor.gr      9523  17.594648s  10.318994s
/net/faraday/workspace/inputs/fl/Hook_1498.gr 14981  24.848609s  13.268091s
/net/faraday/workspace/inputs/fl/Geo_1438.gr  14380  25.694774s   9.703587s
/net/faraday/workspace/inputs/fl/Serena.gr    13914  23.548173s  13.240381s
/net/faraday/workspace/inputs/fl/Flan_1565.gr 15648  24.696646s   4.167168s
/net/faraday/workspace/inputs/fl/bone010.gr    9868  15.132215s   3.552546s
"""

def main():
  for line in InputRaw.split('\n'):
    if not line.strip():
      continue
    toks = line.strip().split()
    filename = toks[0]
    size = toks[1]

    kws = {'run': RunPy, 'input': filename, 'size': size, 'prog': 'exp/apps/sparselu/gsparselu'}
    subprocess.check_call('{run} --no-default-thread -x Threads::-l::1,4:24:4 -x UseIKDG::-a::0,1 -- {prog} -f {input} -n {size} -v 0'.format(**kws), shell=True)

    kws = {'run': RunPy, 'input': filename, 'size': size, 'prog': 'exp/apps/sparselu/sparselu'}
    subprocess.check_call('{run} --no-default-thread -e Threads::OMP_NUM_THREADS::1,4:24:4 -- {prog} -f {input} -n {size} -v 0'.format(kws), shell=True)

    kws = {'run': RunPy, 'input': filename, 'size': size, 'prog': 'exp/apps/sparselu/sparselu-single'}
    subprocess.check_call('{run} --no-default-thread -e Threads::OMP_NUM_THREADS::1,4:24:4 -- {prog} -f {input} -n {size} -v 0'.format(kws), shell=True)

if __name__ == '__main__':
  main()
