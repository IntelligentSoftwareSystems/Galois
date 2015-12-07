#!/usr/bin/env python
# Output most recent last modified time for subdirectory to file
import sys
import os

root = sys.argv[1]
out = sys.argv[2]

mostrecent = 0
for root, dirs, files in os.walk(root):
  for file in dirs + files:
    m = os.path.getmtime(os.path.join(root, file))
    if m > mostrecent:
      mostrecent = m
with open(out, 'w') as f:
  f.write("{0}\n".format(mostrecent))
