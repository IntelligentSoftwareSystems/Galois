#!/usr/bin/env python

import subprocess
import os

RunPy = '%s/scripts/run.py' % (os.environ['w'])

# filename                                     size
InputRaw = """
/net/faraday/workspace/inputs/fl/bone010.gr    9868
/net/faraday/workspace/inputs/fl/boneS10.gr    9149
/net/faraday/workspace/inputs/fl/Flan_1565.gr 15648
/net/faraday/workspace/inputs/fl/audikw_1.gr   9437
/net/faraday/workspace/inputs/fl/inline_1.gr   5038
/net/faraday/workspace/inputs/fl/Emilia_923.gr 9232
/net/faraday/workspace/inputs/fl/ldoor.gr      9523
/net/faraday/workspace/inputs/fl/Hook_1498.gr 14981
/net/faraday/workspace/inputs/fl/Geo_1438.gr  14380
/net/faraday/workspace/inputs/fl/Serena.gr    13914
\'\'                                            200
"""

def main():
  for line in InputRaw.split('\n'):
    if not line.strip():
      continue
    toks = line.strip().split()
    filename = toks[0]
    size = toks[1]

    kws = {'run': RunPy, 'input': filename, 'size': size, 'prog': 'exp/apps/sparselu/gsparselu'}
    subprocess.check_call('{run} --no-default-thread -x Threads::-t::5:40:5 -x UseIKDG::-a::0,1 -- {prog} -f {input} -n {size} -v 0'.format(**kws), shell=True)

    kws = {'run': RunPy, 'input': filename, 'size': size, 'prog': 'exp/apps/sparselu/sparselu'}
    subprocess.check_call('{run} --no-default-thread -e Threads::OMP_NUM_THREADS::5:40:5 -- {prog} -f {input} -n {size} -v 0'.format(**kws), shell=True)

    kws = {'run': RunPy, 'input': filename, 'size': size, 'prog': 'exp/apps/sparselu/sparselu-single'}
    subprocess.check_call('{run} --no-default-thread -e Threads::OMP_NUM_THREADS::5:40:5 -- {prog} -f {input} -n {size} -v 0'.format(**kws), shell=True)

if __name__ == '__main__':
  main()
