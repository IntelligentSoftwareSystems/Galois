#!/usr/bin/python3

import sys
import glob

if __name__ == "__main__":

	application = sys.argv[1]
	input_graph = sys.argv[2]
	size = sys.argv[3]
	minsup = sys.argv[4]
	log_filename = sys.argv[5]
	out_filename = "/net/ohm/export/iss/pangolin-outputs/" + application + "." + input_graph 
	if application != "tc":
		out_filename = out_filename + "." + size
	if application == "fsm":
		out_filename = out_filename + "." + minsup
	
	log = open(log_filename, 'r')
	if application == "motif":
		res = open("result.txt", "w")
		for line in log:
			if line.find("triangles")!=-1 or line.find("wedges")!=-1 or line.find("4-paths")!=-1 or line.find("3-stars")!=-1 or line.find("4-cycles")!=-1 or line.find("tailed-triangles")!=-1 or line.find("diamonds")!=-1 or line.find("4-cliques")!=-1:
				#print(line.split(' ')[-1], file=res)
				res.write(line.split(' ')[-1])
		res.close()
		same = True
		with open(out_filename) as out, open("result.txt") as res:
			for l1, l2 in zip(out, res):
				if l1 != l2:
					same = False
					print("truth is " + l1 + ", but your answer is " + l2)
					break
		if same:
			print("SUCCESS\n")
	else:
		num = 0
		for line in log:
			if line.find("total_num")!=-1:
				num = int(line.split(' ')[-1])
				#print(num)
		log.close()

		out = open(out_filename, "r")
		for line in out:
			truth = int(line)
			if num == truth:
				print("SUCCESS\n")
			else:
				print("truth is " + str(truth) + ", but your answer is " + str(num))
	out.close()

