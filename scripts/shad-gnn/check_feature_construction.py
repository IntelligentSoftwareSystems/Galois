import csv

"""
@autor: Hochan Lee (hochan.lee@amd.com)

Requirement:

The below two files should exist on the directory
where this script runs.

1) solution.csv is the solution file.
2) 2hop.[host id].feat is the results of the feature construction
that we want to check correctness.

Command:
python check_feature_construction.py

"""
num_hosts = 4

solution = {}
with open("solution.csv", "r") as f:
  reader = csv.reader(f)
  for row in reader:
    rlen = len(row)
    feat = []
    for i in range(1, rlen):
      feat.append(int(row[i]))
    solution[row[0]] = feat

fail = False
for i in range(0, num_hosts):
  with open("2hop."+str(i)+".feat", "r") as f:
    reader = csv.reader(f)
    for row in reader:
      rlen = len(row)
      feat = []
      for j in range(1, rlen):
        feat.append(int(row[j]))
      key = row[0]

      solution_feat = solution[key]
      for j in range(0, rlen-1):
        if solution_feat[j] != feat[j]:
            print(key, " failed at ", j, " on host:", i)
            fail = True

if fail:
  print("Verification failed")
else:
  print("Verification succeeded")
