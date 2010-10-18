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
int numEdges;
Graph* config;
GNode sink;
GNode source;

#include "Builder.h"
#include "Support/ThreadSafe/simple_lock.h"
#include "Support/ThreadSafe/TSStack.h"

threadsafe::ts_stack<GNode> wl;
int threads = 1;

using namespace std;

const int ALPHA=6;
const int BETA=12;

// gatAtSerial in cpp
void gapAtSerial(vector<int>& gapYet,  int h) {
 for(Graph::active_iterator ii = config->active_begin(), ee = config->active_end(); ii != ee; ++ii) {
     Node& node = ii->getData();
             if (node.isSink || node.isSource)
               continue;
           if (h < node.height && node.height < numNodes)
               node.height = numNodes;
     }

     if (&gapYet != NULL) {
       for (int i = h + 1; i < numNodes; i++) {
         gapYet[h]=0;
       }
     }
   }




















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


  for(Graph::active_iterator ii = config->active_begin(), ee = config->active_end(); ii != ee; ++ii) 
   {
	Node& n = ii->getData();
	if(n.isSink)
	{
		sink = *ii;   
	}
	else if(n.isSource)
	{
		source= *ii;
	}
   }

int globalRelabelInterval = numNodes * ALPHA + numEdges;

// something like this has to be done final Counter<GNode<Node>>[] gapYet = new Counter[numNodes];

	vector<int> gapYet(numNodes,0);
     for (int i = 0; i < numNodes; i++) {
       int height = i;
       gapYet[height] =0;
	gapAtSerial(gapYet, height);
     }
int relabelYet = globalRelabelInterval;   //the trigger function has to be implemented....






}





//  Func1 implemented in cpp

initializePreflow(final Counter<GNode>[] gapYet) {  //the parameters will change....


	for(Graph::active_iterator ii = config->active_begin(), ee = config->active_end(); ii != ee; ++ii) {
		Node& element = ii->getData();
		if (!node.isSink && !node.isSource) {
			incrementGap(gapYet, node.height);  //have to implement incrementGap func...
		}
	}


	for (Graph::neighbor_iterator ii = graph->neighbor_begin(source), ee = graph->neighbor_end(source); ii != ee; ++ii) {
		GNode neighbor = *ii;
		Edge& edgeData = config->getEdgeData(source, neighbor);

		int cap = edgedata.cap;
		reduceCapacity(source, neighbour, cap);    
		Node& node = neighbour->getData();
		node.excess += cap;
		if (cap > 0)
			wl.push(neighbour);
	}

}

/*
//Func2 implemented in cpp
  private static void decrementGap(final Counter<GNode>[] gapYet, ForeachContext<GNode<Node>> ctx, int height) {
    if (gapYet != null) {
      gapYet[height].increment(ctx, MethodFlag.NONE);   //need to implement this increment function provided by Counter interface
    }
  }


//Func3 implemented in cpp
  private static void incrementGap(final Counter<GNode<Node>>[] gapYet, ForeachContext<GNode<Node>> ctx, int height) {
    if (gapYet != null) {
      gapYet[height].increment(ctx, -1, MethodFlg.ALL);  //need to implement this increment function provided by Counter class 
    }
  }


//Func4 implemented in cpp
  private static void incrementGap(final Counter<GNode>[] gapYet, int height) {
    if (gapYet != null) {
      gapYet[height].increment(-1);   //need to implement increment function of Counter interface
    }
  }


//Func5 implemented in cpp
  private void reduceCapacity(GNode src, GNode dst, int amount) {
    Edge e1 = config->getEdgeData(src, dst);
    Edge e2 = config->getEdgeData(dst, src);
    e1.cap -= amount;
    e2.cap += amount;
  }



bool discharge( threadsafe::ts_stack<GNode>& lwl , Counter<GNode>[] gapYet, GNode src) {
	Node& node = src.getData();
	int prevHeight = node.height;
	bool retval = false;

	if (node.excess == 0 || node.height >= numNodes)
		return ;

	Local l;
	l.src = src;

	while (true) {
		l.finished = false;

		//src.map(dbody, l, ctx, !retval ? MethodFlag.CHECK_CONFLICT : MethodFlag.NONE);  // this map function has to be edited...

		for (Graph::neighbor_iterator ii = graph->neighbor_begin(src), ee = graph->neighbor_end(src); ii != ee; ++ii) {
			GNode dst=*ii;
			if (l.finished)
				break;
			Node& node = l.src.getData();

			int cap = config->getEdgeData(l.src, dst).cap;  //this needs to be refined

			if (cap > 0 && l.cur >= node.current) {
				int amount = 0;
				Node& dnode = dst.getData();
				if (node.height - 1 == dnode.height) {
					// Push flow
					amount = (int) Math.min(node.excess, cap);
					reduceCapacity(l.src, dst, amount);
					// Only add once
				}
				node.excess -= amount;
				dnode.excess += amount;
			}
		}


		if (l.finished)
			break;

		// Went through all our edges and still
		// have flow: Relabel
		relabel(src, l);

		retval = true;
		decrementGap(gapYet, ctx, prevHeight); //check new implementation...

		if (node.height == numNodes)
			break;

		incrementGap(gapYet, ctx, node.height); //check new implemention ....
		prevHeight = node.height;
		l.cur = 0;
	}

	return retval;
}



//Func relabel implemented in cpp

  private void relabel(GNode src, Local& l) {
    l.resetForRelabel();
    
src.map(rbody, src, l, MethodFlag.NONE);   // this map function has to be edited....

    l.minHeight++;

    Node& node = src.getData();
    if (l.minHeight < numNodes) {
      node.height = l.minHeight;
      node.current = l.minEdge;
    } else {
      node.height = numNodes;
    }
  }


*/
