#include <iostream>
#include <set>
#include <stack>
#include <sys/time.h>
#include <limits.h>
#include <math.h>
#include <string.h>
#include <cassert>
#include "include.h"   //containds Global variables and initialization

using namespace std;

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


void reduceCapacity(GNode& src, GNode& dst, int amount) {
        Edge& e1 = config->getEdgeData(src, dst);
        Edge& e2 = config->getEdgeData(dst, src);
        e1.cap -= amount;
        e2.cap += amount;
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
		<< "Using the preflow push technique\n"
		<< "http://iss.ices.utexas.edu/lonestar/preflowpush.html\n"
		<< "\n";

	config = new Graph();   //config is a variable of type Graph*
	Builder b;	//Builder class has method to read from file and construct graph
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

	cout<<"Step 1 done sucessfully...numNodes is "<< numNodes <<endl;   //debug code

	gapYet.resize(numNodes,0);	
	for (int i = 0; i < numNodes; i++) {
		int height = i;
		gapYet[height] =0;
		gapAtSerial(gapYet, height);
	}
	relabelYet = globalRelabelInterval;   //the trigger function has to be implemented....

	cout<<"Step 2 done sucessfully"<<endl;    //debug code
	initializePreflow(gapYet);

	cout<<"Step 3 done sucessfully"<<endl;    //debug code
	int debug_var=0;
	while (wl.size()) {
		cout<<++debug_var;
		bool suc;
		GNode N = wl.pop(suc);
		process(N, wl);
	}
}


void process(GNode item, threadsafe::ts_stack<GNode>& lwl) {
	int increment = 1;
	if (discharge(lwl, gapYet, item)) {
		increment += BETA;
	}

	relabelYet+=increment;
	if(relabelYet >= globalRelabelInterval) 
		globalRelabelSerial(gapYet,lwl);
}



void  initializePreflow(vector<int>& gapYet) {  


        for(Graph::active_iterator ii = config->active_begin(), ee = config->active_end(); ii != ee; ++ii) {
                Node& node = ii->getData();
                if (!node.isSink && !node.isSource) {
                        gapYet[node.height]--;  
                }
        }


        for (Graph::neighbor_iterator ii = config->neighbor_begin(source), ee = config->neighbor_end(source); ii != ee; ++ii) {
                GNode neighbor = *ii;
                Edge& edgeData = config->getEdgeData(source, neighbor);

                int cap = edgeData.cap;
                reduceCapacity(source, neighbor, cap);
                Node& node = neighbor.getData();
                node.excess += cap;
                if (cap > 0)
                        wl.push(neighbor);
        }

}




bool discharge(threadsafe::ts_stack<GNode>& lwl , vector<int>& gapYet, GNode& src) {
	Node& node = src.getData();
	int prevHeight = node.height;
	bool retval = false;

	if (node.excess == 0 || node.height >= numNodes)
		return retval;

	Local l;
	l.src = src;

	while (true) {
		l.finished = false;

		for (Graph::neighbor_iterator ii = config->neighbor_begin(src), ee = config->neighbor_end(src); ii != ee; ++ii) {
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
					node.excess < cap ? amount = node.excess : amount = cap;
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
		gapYet[prevHeight]++; //check new implementation...

		if (node.height == numNodes)
			break;

		gapYet[node.height]--; //check new implemention ....
		prevHeight = node.height;
		l.cur = 0;
	}

	return retval;
}




void relabel(GNode& src, Local& l) {
	l.resetForRelabel();

	for (Graph::neighbor_iterator ii = config->neighbor_begin(src), ee = config->neighbor_end(src); ii != ee; ++ii){
		GNode dst = *ii;
		int cap = config->getEdgeData(src, dst).cap; //refined? check Graph.h file if that is the exact function
		if (cap > 0) {
			Node& dnode = dst.getData();
			if (dnode.height < l.minHeight) {
				l.minHeight = dnode.height;
				l.minEdge = l.relabelCur;
			}
		}
		l.relabelCur++;
	}

	l.minHeight++;

	Node& node = src.getData();
	if (l.minHeight < numNodes) {
		node.height = l.minHeight;
		node.current = l.minEdge;
	} else {
		node.height = numNodes;
	}
}




void globalRelabelSerial(vector<int>& gapYet, threadsafe::ts_stack<GNode>& lw) {
	set<GNode> visited;
	deque<GNode> queue;

	for(Graph::active_iterator ii = config->active_begin(), ee = config->active_end(); ii != ee; ++ii) {
		Node& node = ii->getData();
		// Max distance
		node.height = numNodes;
		node.current = 0;

		if (node.isSink) {
			node.height = 0;
			queue.push_front(*ii); 
			visited.insert(*ii);
		}
	}

	if ( &gapYet != NULL) {
		for (int i = 0; i < numNodes; i++) {
			gapYet[i]=0;
		}
	}

	// Do walk on reverse *residual* graph!
	while (!queue.empty()) {
		GNode& src = queue.front();
		queue.pop_front(); //pollFirst();
		for (Graph::neighbor_iterator ii = config->neighbor_begin(src), ee = config->neighbor_end(src); ii != ee; ++ii){
			GNode dst=*ii;        	
			if (visited.find(dst) != visited.end())
				continue;
			Edge& edge = config->getEdgeData(dst,src);
			if (edge.cap > 0) {
				visited.insert(dst);
				Node& node = dst.getData();
				Node& node2 = src.getData();	          
				node.height = node2.height + 1;
				queue.push_back(dst); 
			}
		}
	}

	for(Graph::active_iterator ii = config->active_begin(), ee = config->active_end(); ii != ee; ++ii) {
		Node& node = ii->getData();

		if (node.isSink || node.isSource || node.height >= numNodes) {
			continue;
		}

		if (node.excess > 0) {
			lw.push(*ii);
		}

		gapYet[node.height]--;
	}

}




