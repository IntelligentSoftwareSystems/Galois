#include <iostream>
#include <stack>
#include <sys/time.h>
#include <limits.h>
#include <math.h>
#include <string.h>
#include <cassert>

#include "Galois/Launcher.h"
#include "Galois/Graphs/Graph.h"
#include "Node.h"
#include "Edge.h"



typedef FirstGraph<Node,Edge>            Graph;
typedef FirstGraph<Node,Edge>::GraphNode GNode;


int numNodes;
Graph* config;
GNode sink;
GNode source;

#include "Builder.h"
#include "Support/ThreadSafe/simple_lock.h"
#include "Support/ThreadSafe/TSStack.h"

threadsafe::ts_stack<GNode> wl;
int threads = 1;

using namespace std;

int main(int argc, char** argv) {
  if (argc < 2) {
    cerr << "Arguments: <input file>\n";
    return 1;
  }

  cerr << "\nLonestar Benchmark Suite v3.0\n"
    << "Copyright (C) 2007, 2008, 2009, 2010 The University of Texas at Austin\n"
    << "http://iss.ices.utexas.edu/lonestar/\n"
    << "\n"
    << "application: Preflow Push Algorithm (c++ version)\n"
    << "Finds the maximum flow in a given network\n"
    << "using the preflow push technique\n"
    << "http://iss.ices.utexas.edu/lonestar/preflowpush.html\n"
    << "\n";

  config = new Graph();
  Builder b;
  b.read(config, argv[1]);


/*  for(Graph::active_iterator ii = config->active_begin(), ee = config->active_end(); ii != ee; ++ii) {
	  Node& n = ii->getData();
	if(n.isSink)
	{
		sink =(GNode) n;
	}
	else if(n.isSource)
	{
		source=(GNode)n;
	}
}*/








}

