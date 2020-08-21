#!/usr/bin/python3
import sys
import os
import re

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

if len(sys.argv) != 3:
    print("Invalid arguments: expect <folder_name> <input_name>")
input_folder = sys.argv[1]

content = "BSD 0\n"
host_counter = 0

for filename in sorted([name for name in os.listdir(input_folder)], key=natural_keys):
    print("read file " + filename)
    with open(os.path.join(input_folder, filename), 'r') as f:
        lines = [line.rstrip('\n') for line in f]

        local_stats = ""

        local_stats += f"# {host_counter} {host_counter}\n"

        STR_RD = int(lines[3].split(":")[1])
        local_stats += f"STR RD {STR_RD} {STR_RD*8}\n"

        RND_RD = int(lines[4].split(":")[1])
        local_stats += f"RND_RD {RND_RD} {RND_RD*8}\n"

        RND_WR = int(lines[5].split(":")[1])
        local_stats += f"RND_WR {RND_WR} {RND_WR*8}\n"


        remote_line_offset = 8
        for remote_host in range(0, len(lines) - remote_line_offset):
            if remote_host == host_counter:
                content += local_stats
            else:
                REMOTE_STR_RD = int(lines[remote_host + remote_line_offset].split(":")[1])
                content += f"# {host_counter} {remote_host}\n"
                content += f"STR RD {REMOTE_STR_RD} {REMOTE_STR_RD*8}\n"

    host_counter += 1


with open(f"GAL_WF1_{host_counter}_0_{sys.argv[2]}.stats", "w") as f:
    f.write(content)
