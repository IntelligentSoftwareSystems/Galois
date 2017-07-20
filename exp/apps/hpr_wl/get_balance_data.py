import sys

filename = sys.argv[1]

filedata = open(filename, "r")
data = filedata.readlines()
filedata.close()

data = {}

for line in data:
    if "debug" in line:
        line = line.split(",")
        if int(line[1]) not in data:
            data[int(line[1])] = []
        data[int(line[1])].append(int(line[3]))

print data