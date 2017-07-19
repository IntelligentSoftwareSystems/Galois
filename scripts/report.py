#!/usr/bin/env python

import sys
import collections

def main():
  class Row:
    def __init__(self):
      self.reset()
    def reset(self):
      self.r = collections.defaultdict(str)
      self.header = None
    def get(self, token, key):
      return token[self.header.index(key)]

  row = Row()
  rows = []
  cols = set()

  for line in sys.stdin:
    try:
      param_token = [i.strip() for i in line.split()]
      stat_token = [i.strip() for i in line.split(",")]

      # empty line
      if param_token == []:
        continue

      # parameter setting by run.py
      if param_token[0] == "RUN:":
        if param_token[1] == "Start":
          if row.r:
            rows.append(row.r)
            row.reset()
        elif param_token[1] == "Variable":
          key = param_token[2]
          cols.add(key)
          row.r[key] = param_token[4] # param_token[3] is "="
        elif param_token[1] == "CommandLine":
          cmd_token = [i.strip() for i in line.split(None, 2)]
          key = cmd_token[1]
          cols.add(key)
          row.r[key] = cmd_token[2]

      # stat header returned by Galois
      elif stat_token[0] == "LOOP":
        row.header = stat_token

      # stat lines. ignore HOST for shared-memory version
      elif row.header != None:
        loop_name = row.get(stat_token, "LOOP")
        instance = row.get(stat_token, "INSTANCE")
        th = row.get(stat_token, "THREAD")
        key = row.get(stat_token, "CATEGORY") + "-t" + th
        if loop_name != "(NULL)":
          key = loop_name + "-i" + instance + "-" + key
        cols.add(key)
        row.r[key] = row.get(stat_token, "VAL")

    except:
      sys.stderr.write("Error parsing line: %s" % line)
      raise

  if row.r:
    rows.append(row.r)
  cols = sorted(cols)

  print(','.join(cols))
  for r in rows:
    print(','.join([str(r[c]) for c in cols]))


if __name__ == "__main__":
  main()
