#!/usr/bin/env python
# Output wheel configuration variables to stdout

import distutils.util
import sys

mode = sys.argv[1]

if mode == 'platform':
  print(distutils.util.get_platform().replace('-', '_').replace('.', '_'))
elif mode == 'python':
  print('cp{0}'.format(''.join([str(x) for x in sys.version_info[0:2]])))
