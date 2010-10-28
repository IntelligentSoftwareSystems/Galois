/*
 * main.cpp
 *
 *  Created on: Oct 26, 2010
 *      Author: amshali
 */

#include "SSSP.h"
#include <iostream>
#include <fstream>
using namespace std;

int main(int argc, char *argv[]) {
	Graph* g = NULL; //config is a variable of type Graph*
	char* inputfile = NULL;
	bool bfs;
	if (argc != 3) {
		cout << "Usage: <bfs> <input-file>" << endl;
		exit(-1);
	} else {
		bfs = strcmp(argv[1], "f") == 0 ? false : true;
		inputfile = argv[2];
	}
	ifstream infile;
	infile.open(inputfile, ifstream::in); // opens the vector file
	if (!infile) { // file couldn't be opened
		cerr << "Error: vector file could not be opened" << endl;
		exit(-1);
	}

	string name;
	int numNodes, numEdges;
	SNode **nodes;
	GNode *gnodes;
	while (!infile.eof()) {
		string line;
		string firstchar;
		infile >> firstchar;
		if (!strcmp(firstchar.c_str(), "c")) {
		} else if (!strcmp(firstchar.c_str(), "p")) {
			infile >> name;
			infile >> numNodes;
			infile >> numEdges;
			g = new Graph();
			nodes = new SNode*[numNodes];
			gnodes = new GNode[numNodes];
			for (int i = 0; i < numNodes; i++) {
				nodes[i] = new SNode(i+1);
				gnodes[i] = g->createNode(*nodes[i]);
				g->addNode(gnodes[i]);
			}
			cout << "graph name is " << name << " and it has " << numNodes
					<< " nodes and some edges" << endl;
		} else if (!strcmp(firstchar.c_str(), "a")) {
			int src, dest, weight;
			infile >> src >> dest >> weight;
			g->addEdge(gnodes[src-1], gnodes[dest-1], SEdge(weight));
			cout << "node: " << src << " " << dest << " " << weight << endl;
		}
		getline(infile, line);
	}
	infile.close();
	exit(0);
}
