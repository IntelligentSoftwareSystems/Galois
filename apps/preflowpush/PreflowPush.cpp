#include <iostream>
#include <stack>
#include <sys/time.h>
#include <limits.h>
#include <math.h>
#include <string.h>
#include <cassert>
#include "include.h"   //containds Global variables and initialization
#include "Lonestar/Banner.h"
#include "Lonestar/CommandLine.h"

int counting=0;
using namespace std;

static const char* name = "Preflow Push";
static const char* description = "Finds the maximum flow in a given network using the preflow push technique\n";
static const char* url = "http://iss.ices.utexas.edu/lonestar/preflowpush.html";
static const char* help = "<input file> <expected flow>";


void reduceCapacity(GNode& src, GNode& dst, int amount) {
        Edge& e1 = config->getEdgeData(src, dst,Galois::Graph::ALL,0);
        Edge& e2 = config->getEdgeData(dst, src,Galois::Graph::ALL,0);
        e1.cap -= amount;
        e2.cap += amount;
}


bool check(int h)
{
  int i;
  if(gapYet[h]==0)
    {
      for(i=h+1;i<(int)gapYet.size();i++)
	if(gapYet[i]>0)
	  return true;
    }
  return false;
}


void checkHeight(int h){
    for(Graph::active_iterator ii = config->active_begin(), ee = config->active_end(); ii != ee; ++ii) {
	    Node& node = (*ii).getData(Galois::Graph::ALL,0);
	    if (node.isSink || node.isSource)
		    continue;
	    if (h < node.height && node.height < numNodes)
		    node.height = numNodes;
    }
}


void printHeights()
{
	for(Graph::active_iterator ii = config->active_begin(), ee = config->active_end(); ii != ee; ++ii) {
		Node& node = (*ii).getData(Galois::Graph::ALL,0);
		cout<<" Inside printHeight func.. node.height = "<<node.height<<endl;
	}
}

struct process {
  template<typename Context>
  void operator()(GNode& item, Context& lwl) {
/*	bool inRelabel = false;
	try
	{
		while(lock==true);
		__sync_fetch_and_add(&counter,1);
		int increment = 1;*/
		if(discharge(item, &lwl));
		/*	increment += BETA;
		__sync_fetch_and_add(&relabelYet, increment);
		if(relabelYet > globalRelabelInterval)
		{	
			if( __sync_val_compare_and_swap(&lock, false, true) == false)
			{
				flag = 0;
				//cout<<"Trying Global Relabel..."<<endl;
				while(counter!=1);
				inRelabel = true;
				//cout<<"Doing Global Relabel...Counter is "<<counter<<endl;
				globalRelabelSerial(&lwl);
				counting++;
				__sync_val_compare_and_swap(&relabelYet, relabelYet, 0);
				//cout<<"Global relabel complete... "<<counter<<endl;
				flag = 1;
				lock = false;
			}
		}
		__sync_fetch_and_add(&counter, -1);
		//cout<<"Counter at end :"<<counter<<endl;
	}catch(...)
	{
		if (inRelabel)
		{
			//cout<<"Exception in global relabel"<<endl;
			__sync_val_compare_and_swap(&lock, true, false);
			flag = 1;
		}
		__sync_fetch_and_add(&counter, -1);
		throw;	
	}*/
  }
};

int main(int argc, const char** argv) {

  std::vector<const char*> args = parse_command_line(argc, argv, help);

  if (args.size() < 1) {
    std::cout << "not enough arguments, use -help for usage information\n";
    return 1;
  }
  printBanner(std::cout, name, description, url);

  if (args.size() > 1)
    expected=atoi(args[1]);
  else   //if expected flow value is not provided 
    {
      cout<<"!! Expected flow value not provided as arguement !!\n";
      cout << "Arguments: <input file> <expected flow value>\n";
      cout << "Assuming 0\n";
      expected=0;
    }


	config = new Graph();   //config is a variable of type Graph*
	Builder b;	//Builder class has method to read from file and construct graphb.read_wash(config, argv[inputFileAt]);
	b.read_rand(config, args[0]);
	cout<<"numNodes is "<< numNodes <<endl;   //debug code

	gapYet.resize(numNodes+1,0);	
	//wl.resize(numNodes);
	//initialize gapYet
	for(Graph::active_iterator ii = config->active_begin(), ee = config->active_end(); ii != ee; ++ii) {
		Node& node = (*ii).getData();
		gapYet[node.height]++;
	}


	globalRelabelInterval = numNodes * ALPHA + numEdges;

	cout<<"Global Relabel = "<<globalRelabelInterval<<endl;
	relabelYet = 0;   //the trigger function has to be implemented....

	initializePreflow();
	cout<<"Size of worklist is : "<<wl.size()<<endl;
	cout<<"Source and sink are "<<source.getData().id<<"   "<<sink.getData().id<<endl;
	//int increment=1;	
	Galois::Launcher::startTiming();
	/*while (wl.size()) {
	  bool suc;
	  GNode N = wl.pop(suc);
	//process(N);		
	if(discharge(N))
		increment+=BETA;
	if(increment > globalRelabelInterval)
	{
		cout<<"Doing Global Relabel"<<endl;
		globalRelabelSerial();		
		increment=0;
	}
	}*/
	Galois::setMaxThreads(numThreads);
	cout<<"Threads :"<<threads<<endl;
	Galois::for_each(wl.begin(), wl.end(), process());
	Galois::Launcher::stopTiming();

	cout<<"Number of global relabels is :"<<counting<<endl;	
	cout<<"Flow is "<<sink.getData().excess<<endl;
	//checkMaxFlow();
	cout<<"Flow is OK"<<endl;
}


void  initializePreflow() {  
  //	int count=0;
	for (Graph::neighbor_iterator ii = config->neighbor_begin(source), ee = config->neighbor_end(source); ii != ee; ++ii) {
		GNode neighbor = *ii;
		Edge& edgeData = config->getEdgeData(source, neighbor,Galois::Graph::NONE,0);

		int cap = edgeData.cap;
		reduceCapacity(source, neighbor, cap);
		Node& node = neighbor.getData();
		node.excess += cap;
		if (cap > 0)
		{	
			node.insert_time = ++token;
			wl.push_back(*ii);
		}
	}

}



template<typename Context>
bool discharge(GNode& src, Context* cnx) {
	Node& node = src.getData(Galois::Graph::ALL);
	int prevHeight = node.height;
	//cout<<"Height of node is "<<prevHeight<<"  Insert Time is : "<<node.insert_time<<endl;
	bool retval = false;

	if (node.excess == 0 || node.height >= numNodes)
		return retval;
	//	static int i=0;
	//	int j=0;	
	Local l;
	l.src = src;
	while (true) {
		l.finished = false;

		for (Graph::neighbor_iterator ii = config->neighbor_begin(src), ee = config->neighbor_end(src); ii != ee; ++ii) {
			GNode dst=*ii;
			Node& node2 = l.src.getData(Galois::Graph::ALL);
			Node& dnode = dst.getData(Galois::Graph::ALL);
			if (l.finished)
				break;
			int cap = config->getEdgeData((l.src), dst,Galois::Graph::ALL).cap;  //this needs to be refined
			if (cap > 0 && l.cur >= node2.current) {
				int amount = 0;
				if (node2.height - 1 == dnode.height) {
					// Push flow
					node2.excess < cap ? amount = node2.excess : amount = cap;
					reduceCapacity((l.src), dst, amount);
					// Only add once
					if (!dnode.isSink && !dnode.isSource && dnode.excess == 0) {
						//cout<<"Adding to list..."<<endl;
						__sync_fetch_and_add(&token, 1);
						dnode.insert_time=token;
						cnx->push(dst);
					}

					node2.excess -= amount;
					dnode.excess += amount;
					if (node2.excess == 0) {
						l.finished = true;
						node2.current = l.cur;						
						break;
					}
				}
			}
			l.cur++;
		}

		if (l.finished)
			break;
		// Went through all our edges and still have flow: Relabel
		relabel(src, l, cnx);
		retval = true;
		gapYet[prevHeight]--;
		gapYet[node.height]++;
		if(check(prevHeight))
			checkHeight(prevHeight);

		if (node.height == numNodes)
			break;

		prevHeight = node.height;
		l.cur = 0;
	}
	return retval;
}


template<typename Context>
void relabel(GNode& src, Local& l, Context* cnx) {
	l.resetForRelabel();

	for (Graph::neighbor_iterator ii = config->neighbor_begin(src), ee = config->neighbor_end(src); ii != ee; ++ii){
		GNode dst = *ii;
		int cap = config->getEdgeData(src, dst,Galois::Graph::ALL).cap; //refined? check Graph.h file if that is the exact function
		if (cap > 0) {
			Node& dnode = dst.getData(Galois::Graph::ALL);
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



template<typename Context>
void globalRelabelSerial  (Context* cnx )  
{
	//vector<GNode> visited;
	//deque<GNode> queue;
	//usleep(50000);
	for(Graph::active_iterator ii = config->active_begin(), ee = config->active_end(); ii != ee; ++ii) {
		Node& node = ii->getData(Galois::Graph::ALL);
		// Max distance
		node.height = numNodes;
		node.current = 0;
	}
	Node& temp=sink.getData(Galois::Graph::ALL);
	temp.height = 0;	
	que.push(sink);
//cout<<"End of Phase 1"<<endl; 

	gapYet.assign(numNodes+1,0);
	//Galois::setMaxThreads(threads);
	//Galois::for_each(que,bfs);
	while (!que.empty()) {
		std::pair<bool, GNode> ret = que.pop(); //pollFirst();
		GNode src = ret.second;
		for (Graph::neighbor_iterator ii = config->neighbor_begin(src), ee = config->neighbor_end(src); ii != ee; ++ii){
			GNode dst=*ii;        	
			Edge& edge = config->getEdgeData(dst,src,Galois::Graph::ALL);
			if (edge.cap > 0) {
				Node& node = dst.getData(Galois::Graph::ALL);
				int newHeight=src.getData(Galois::Graph::ALL).height+1;
				if(newHeight<node.height){
					node.height = newHeight;
					que.push(dst);
				}
			}
		}
	}

//cout<<"End of Phase 2"<<endl; 
	for(Graph::active_iterator ii = config->active_begin(), ee = config->active_end(); ii != ee; ++ii) {
		GNode temp=*ii;
		Node& node = ii->getData(Galois::Graph::ALL);

		if (node.isSink || node.isSource || node.height >= numNodes) {
			continue;
		}

		gapYet[node.height]++;

		if (node.excess > 0) {
			cnx->push(temp);
		}		
	}


	cout<<"Global Relabel ends..Flow is "<<sink.getData(Galois::Graph::ALL).excess<<endl;

}


template<typename Context>
void bfs(GNode& src ,Context* cnx)
{
	for (Graph::neighbor_iterator ii = config->neighbor_begin(src), ee = config->neighbor_end(src); ii != ee; ++ii){
		//	cout<<"Inside BFS..."<<endl;
		GNode dst=*ii;
		//if (  visited.search(dst) )
		//        continue;
		Edge& edge = config->getEdgeData(dst,src,Galois::Graph::ALL);
		if (edge.cap > 0) {
			//visited.push(dst);
			Node& node = dst.getData(Galois::Graph::ALL);
			//Node& node2 = src.getData();
			int newHeight=src.getData(Galois::Graph::ALL).height+1;
			if(newHeight<node.height){
				node.height = newHeight;
				que.push(dst);
			}
		}
	}

}

/*
void checkMaxFlow() {
	double result=sink.getData().excess;
	if ( expected == result ) {
		checkFlowsForCut();
		checkHeights();
		checkAugmentingPathExistence();
	} else {
		if (result != expected) {
			cerr<<"Inconsistent flows: "<<expected<<" != "<<result<<endl;
			exit(-1);
		}
	}
}


void checkFlowsForCut() 
{
	// Check conservation
	for(Graph::active_iterator ii = config->active_begin(), ee = config->active_end(); ii != ee; ++ii) 
	{
		GNode src=*ii;
		Node& node = src.getData();
		if (node.isSource || node.isSink) 
			continue;

		if (node.excess < 0)
		{cout<<"Excess at "<<node.id;exit(-1);}

		double sum;
		for (Graph::neighbor_iterator jj = config->neighbor_begin(src), ff = config->neighbor_end(src); jj != ff; ++jj) 
		{
			GNode dst = *jj;


			int ocap = config->getEdgeData(src, dst,Galois::Graph::ALL,0).ocap;
			int delta = 0;
			if (ocap > 0) 
			{
				delta -= ocap - config->getEdgeData(src, dst,Galois::Graph::ALL,0).cap;

			} else 
			{
				delta += config->getEdgeData(src, dst,Galois::Graph::ALL,0).cap;
			}
			sum+=delta;
		}

		if (node.excess != sum)
		{	cerr<<"Not pseudoflow "<<node.excess<<" != "<<sum<<" at node "<<node.id<<endl;exit(-1);}
	}
}


void checkHeights() {
	for(Graph::active_iterator ii = config->active_begin(), ee = config->active_end(); ii != ee; ++ii)
	{
		GNode src=*ii;
		for (Graph::neighbor_iterator jj = config->neighbor_begin(src), ff = config->neighbor_end(src); jj != ff; ++jj)
		{
			GNode dst = *jj;
			int sh = src.getData().height;
			int dh = dst.getData().height;
			int cap = config->getEdgeData_directed(src, dst).cap;
			if (cap > 0 && sh > dh + 1) 
			{	cout<<"height violated "<<endl;exit(-1);}
		}
	}
}




void checkAugmentingPathExistence() {
	vector<GNode> visited;
	deque<GNode> queue;

	visited.push_back(source);
	queue.push_back(source);

	while (!queue.empty()) {
		GNode& src = queue.front();
		queue.pop_front(); //pollFirst();
		for (Graph::neighbor_iterator ii = config->neighbor_begin(src), ee = config->neighbor_end(src); ii != ee; ++ii)
		{
			GNode dst=*ii;
			if (  (find( visited.begin(), visited.end(), (dst) ) == visited.end()) && config->getEdgeData(src, dst,Galois::Graph::ALL,0).cap > 0  )
			{
				visited.push_back(dst);
				queue.push_back(dst);
			}


			if (find( visited.begin(), visited.end(), (dst) ) != visited.end()) 
			{ 
				cout<<"Augmenting path exists"<<endl;
				exit(-1); 
			}

		}
	}
}
*/
