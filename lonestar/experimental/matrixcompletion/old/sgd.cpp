/* 
 * License:
 *
 * Copyright (C) 2012, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 *
 * Stochastic gradient descent for matrix factorization, implemented with Distributed Galois.
 * 
 * Author: Gurbinder Gill <gill@cs.utexas.edu>
 * Author: Bharat Naik    <bharatpn@cs.utexas.edu>
 * Author: Prad Nelluru <pradn@cs.utexas.edu>
*/

#include <iostream>
#include <random>
#include <cmath>
#include <algorithm>
#include <utility>
#include <unordered_map>
#include <map>
#include <atomic>
#include <vector>
#include <cstdint>

#include "Galois/Galois.h"
#include "Galois/Graph/Graph.h"
#include "Galois/Graph/LCGraph.h"
#include "Galois/ParallelSTL/ParallelSTL.h"
#include "llvm//Support/CommandLine.h"
#include "Lonestar/BoilerPlate.h"
#include "Galois/Runtime/Network.h"
#include "Galois/Timer.h"
#include "Galois/Timer.h"
#include "Galois/Graph/FileGraph.h"
//Distributed Galois
#include "Galois/Graphs/Graph3.h"
#include "Galois/Runtime/DistSupport.h"
#include "Galois/Reduction.h"


#include <boost/iterator/transform_iterator.hpp>
#define LATENT_VECTOR_SIZE 2
typedef int EdgeData;
unsigned int num_user_nodes = 0;
uint64_t idCount = Galois::Runtime::NetworkInterface::ID*(num_user_nodes)+1;
typedef struct Node
{
  //double* latent_vector; //latent vector to be learned
  double latent_vector[LATENT_VECTOR_SIZE]; //latent vector to be learned
  uint64_t updates; //number of updates made to this node (only used by movie nodes)
  uint64_t edge_offset; //if a movie's update is interrupted, where to start when resuming
  uint64_t ID;
  uint32_t number_of_edges;
  //iuint64_t hostID;  
  Node() {
      updates=0;
      number_of_edges=0;
      edge_offset=0;
      ID = idCount++;
      unsigned int seed = 42;
      std::default_random_engine eng(seed);
      std::uniform_real_distribution<double> random_lv_value(0, 0.1);
      double* lv = new double[LATENT_VECTOR_SIZE];
      for(int i = 0; i < LATENT_VECTOR_SIZE; i++)
      {
	  lv[i] = random_lv_value(eng);
      }
      for(int i=0;i<LATENT_VECTOR_SIZE;i++) {
	  latent_vector[i] = lv[i];
      }

  } 
    bool operator==(const Node& other) const
    {
	return (ID == other.ID);
    }
    bool operator<(const Node& other) const 
    {
	return (ID < other.ID);
    }

typedef int tt_has_serialize;
    void serialize(Galois::Runtime::SerializeBuffer& s) const {
	gSerialize(s,latent_vector, updates, edge_offset, ID, number_of_edges);
    }
    void deserialize(Galois::Runtime::DeSerializeBuffer& s) {
	gDeserialize(s,latent_vector, updates, edge_offset, ID, number_of_edges);
    } 


} Node;



using std::cout;
using std::endl;
namespace std {
    
    template <>
    struct hash<Node>
    {
	std::size_t operator()(const Node& other) const {
	    using std::hash;
	    using std::size_t;
	    return (hash<unsigned int>()(other.ID));
	}
    };	
}


//local computation graph (can't add nodes/edges at runtime)
//node data is Node, edge data is unsigned int... [movie--->user]

//typedef Galois::Graph::LC_Numa_Graph<Node, unsigned int> Graph;
//typedef Graph::GraphNode GNode;
/*typedef Galois::Graph::FileGraph Graph;
typedef uint64_t GNode;
Graph File_graph;
*/

//Distributed Graph Nodes.
typedef Galois::Graph::ThirdGraph<Node, uint32_t, Galois::Graph::EdgeDirection::Out> DGraph;
typedef DGraph::NodeHandle DGNode;
typedef typename DGraph::pointer Graphp;


typedef Galois::Graph::FileGraph FGraph;
typedef Galois::Graph::FileGraph::GraphNode FileGNode;
FGraph fgraph;

//typedef Galois::Graph::LC_CSR_Graph<Node, unsigned int> Graph;
//typedef Graph::GraphNode GNode;

//Graph graph;

// TODO : replace maps with unordered_map
/*std::unordered_map<GNode,Node> lookup;
std::unordered_map<GNode,DGNode> mapping;
std::unordered_map<Node,DGNode> llookup;
std::unordered_map<Node,DGNode> rlookup;
std::set<Node> requested;
*/

std::unordered_map<FileGNode,DGNode> mapping;
// DGNode ==> hostID mapping
std::map<DGNode, uint64_t> HostIDMap;

//using accumulator = Galois::GAccumulator<int64_t>;
using accumulator = Galois::DGReducible<int64_t>;

//Processed movie nodes:
uint64_t Processed_movie_nodes = 0;

volatile unsigned prog_barrier = 0;
//std::atomic<unsigned> prog_barrier;
//unsigned int num_movie_nodes = 0;
unsigned int num_movie_nodes = 0;
using namespace Galois::Runtime;
typedef Galois::Runtime::LL::SimpleLock SLock;
SLock slock;
SLock pblock;

//unsigned int LATENT_VECTOR_SIZE = 2;
double LEARNING_RATE = 0.001;
double DECAY_RATE = 0.9;
double LAMBDA = 0.001;
unsigned int MAX_MOVIE_UPDATES = 0;
unsigned int NUM_RATINGS = 0;
static const double MINVAL = -1e+100;
static const double MAXVAL = 1e+100;


double vector_dot(const Node& movie_data, const Node& user_data) {
  const double* __restrict__ movie_latent = movie_data.latent_vector;
  const double* __restrict__ user_latent = user_data.latent_vector;

  double dp = 0.0;
  for (int i = 0; i < LATENT_VECTOR_SIZE; ++i)
    dp += user_latent[i] * movie_latent[i];
    //dp += user_data.latent_vector[i] * movie_data.latent_vector[i];
  assert(std::isnormal(dp));
  return dp;
}

double calcPrediction (const Node& movie_data, const Node& user_data) {
  double pred = vector_dot(movie_data, user_data);
  pred = std::min (MAXVAL, pred);
  pred = std::max (MINVAL, pred);
  return pred;
}

inline void doGradientUpdate(Node& movie_data, Node& user_data, uint64_t edge_rating)
{
  double* __restrict__ movie_latent = movie_data.latent_vector;
        double step_size = LEARNING_RATE * 1.5 / (1.0 + DECAY_RATE * pow(movie_data.updates + 1, 1.5));
  double* __restrict__ user_latent = user_data.latent_vector;

  double cur_error = edge_rating - vector_dot(movie_data, user_data);

  for(unsigned int i = 0; i < LATENT_VECTOR_SIZE; i++)
    {
      double prev_movie_val = movie_latent[i];
      double prev_user_val = user_latent[i];
      movie_latent[i] += step_size * (cur_error * prev_user_val  - LAMBDA * prev_movie_val);
      user_latent[i]  += step_size * (cur_error * prev_movie_val - LAMBDA * prev_user_val);
    }
}


Galois::DGReducible<double> RMS;
Galois::DGReducible<unsigned> count_data;
/*void verify() {
typedef Galois::GAccumulator<double> AccumDouble;
    AccumDouble rms;
    cout<<"Host:"<<Galois::Runtime::NetworkInterface::ID<<" is verifying before SGD..\n";
    //Galois::do_all(graph.begin(), graph.begin()+num_movie_nodes, [&] (GNode n) {
    for(auto ni = graph.begin(), ei = graph.begin()+num_movie_nodes; ni != ei; ++ni) {
	for(auto ii = graph.edge_begin(*ni); ii != graph.edge_end(*ni); ++ii){
	    GNode m = graph.getEdgeDst(ii);
	    double pred = calcPrediction(graph.getData(*ni), graph.getData(m));
	    double rating = graph.getEdgeData(ii);
	    if(!std::isnormal(pred))
		std::cout << "Denormal Warning\n";
	    rms += ((pred - rating)*(pred - rating));

	}

    }
    cout<<"Reached end..\n"<<endl;
    double total_rms = rms.reduce();
    double normalized_rms = sqrt(total_rms/NUM_RATINGS);
    std::cout << "RMSE Total: " << total_rms << " Normalized: " << normalized_rms << std::endl;
}*/
void verify(Graphp g){
    typedef Galois::DGReducible<double> AccumDouble;
    AccumDouble rms;
    cout<<"Host:"<<Galois::Runtime::NetworkInterface::ID<<" is verifying after SGD..\n";
    //Galois::do_all_local(g, [&g,&rms] (DGNode n) {
	auto ei = g->begin();
	std::advance(ei,num_movie_nodes);
	unsigned int count=0;
	for(auto ni = g->begin(); ni != ei; ++ni) {
			for(auto ii = g->edge_begin(*ni); ii != g->edge_end(*ni); ++ii){
			
			    DGNode m = g->getEdgeDst(ii);
			    double pred = calcPrediction(g->getData(*ni), g->getData(m));
			    unsigned int rating = ii->getValue();
			    rating = rating%10;
		
			    if(!std::isnormal(pred))
				std::cout << "Denormal Warning\n";
			    rms += ((pred - rating)*(pred - rating));
				
			}
			count++;
			    
    }
    cout<<"Reached end..\n"<<endl; 
    double total_rms = rms.reduce();
    double normalized_rms = sqrt(total_rms/NUM_RATINGS);
    std::cout << "RMSE Total: " << total_rms << " Normalized: " << normalized_rms << std::endl;
 //   cout<<"Number of nodes seen = "<<count<<endl;

}

void printNode(const Node& t) {
    cout<<"ID: "<<t.ID<<endl;
    cout<<"Edge_offset: "<<t.edge_offset<<endl;
    cout<<"Updates: "<<t.updates<<endl;
    cout<<"Number of edges: "<<t.number_of_edges<<endl;
    for(int i=0;i<LATENT_VECTOR_SIZE;i++) {
        cout<<" "<<t.latent_vector[i]<<endl;
    }
}
/* Operator */
Galois::DGReducible<size_t> numNodes; 
unsigned count_done=0;
struct sgd_algo {
   //unsigned int num_movie_nodes;

struct Process : public Galois::Runtime::Lockable {
    Graphp g;
    sgd_algo* self;
    int iteration;
    unsigned int startRange;
    unsigned int endRange;
    Process(){ }
    // sgd(Graphp _g) : g(_g) {}
    Process(sgd_algo* s, Graphp _g, unsigned int _start, unsigned int _end) : g(_g), self(s), startRange(_start), endRange(_end) { }
    //void operator()(const DGNode& n, Galois::UserContext<DGNode>&) {(*this)(n);} 
    void operator()(const DGNode& movie, Galois::UserContext<DGNode>& ctx)
    {
     Node& movie_data = g->getData(movie);
     //cout <<"ID of movie: "<<movie_data.ID<<endl;
    //printNode(movie_data);

     DGraph::edge_iterator edge_it = g->edge_begin(movie);
     DGraph::edge_iterator edge_end = g->edge_end(movie);
/**********************************************************
* new appraoch.. Process all the edges of a movie node in 
* in one go.
* ********************************************************/ 	
    //uint32_t edges = std::distance(edge_end,edge_it);
   /* 	
    if(movie_data.edge_offset < movie_data.number_of_edges)
	std::advance(edge_it, movie_data.edge_offset);
    else if(movie_data.edge_offset == movie_data.number_of_edges){
	std::advance(edge_it, movie_data.edge_offset);
	movie_data.edge_offset = 0;	
    }
    else 
	movie_data.edge_offset = 0;	
   
    for(auto ii = edge_it; ii != edge_end; ++ii) {
	DGNode user = g->getEdgeDst(edge_it);
	Node& user_data = g->getData(user);
	
	unsigned int edge_rating = edge_it->getValue();
		
		doGradientUpdate(movie_data, user_data, edge_rating);
		++movie_data.edge_offset;
    }
    
    numNodes += 1;
    ++Processed_movie_nodes;
    printf("Processed = %lu\t , hostID = %d\n", Processed_movie_nodes, Galois::Runtime::NetworkInterface::ID);
    if(movie_data.edge_offset == movie_data.number_of_edges)
	++movie_data.updates;
			if(movie_data.updates < MAX_MOVIE_UPDATES)
				ctx.push(movie);

*/

     //std::advance(edge_it,  movie_data.edge_offset);
     //std::advance(edge_it,  startRange);
     //cout<<Galois::Runtime::NetworkInterface::ID<<" checking if multiple edges..\n"<<endl;
     assert(edge_it != edge_end);
     unsigned int edge_rating;
     unsigned int dstID;
     if(edge_it != edge_end) {
	edge_rating = edge_it->getValue();
	dstID = edge_rating/10;
		}
     //cout<<"Iterating till start.."<<endl;
     while(dstID < startRange && edge_it != edge_end){
	edge_rating = edge_it->getValue();
	dstID = edge_rating/10;
	++edge_it;
    }
    //cout<<Galois::Runtime::NetworkInterface::ID<<" reached range start...\n"<<endl;
     //else if(dstID >= endRange)
    //	return;
     while((dstID < endRange) && (edge_it != edge_end)) {

	 edge_rating = edge_it->getValue();
	 dstID = edge_rating/10;
	 DGNode user = g->getEdgeDst(edge_it);
	 Node& user_data = g->getData(user);

	 edge_rating = edge_rating%10;
	 //cout<<"Value of rating: "<<edge_rating<<endl;

	 // Call the gradient routine
	 doGradientUpdate(movie_data, user_data, edge_rating);
	 ++edge_it;
	 //++movie_data.edge_offset;
	 //assert(startRange<=dstID && dstID <= endRange);
     }
   //cout<<Galois::Runtime::NetworkInterface::ID<<" reached range end...\n"<<endl;
	   // This is the last user
     if(edge_it == edge_end)// Galois::MethodFlag::NONE))
     {
	//start back at the first edge again
	movie_data.edge_offset = 0;
	numNodes += 1;
	++Processed_movie_nodes;
	
	//printf("Processed = %lu\t , hostID = %d\n", Processed_movie_nodes, Galois::Runtime::NetworkInterface::ID);
	movie_data.updates++;
	//cout<<"Done with this movie.. count = "<<++count_done<<" host = "<<Galois::Runtime::NetworkInterface::ID<<endl;
	if(movie_data.updates < MAX_MOVIE_UPDATES)
	    ctx.push(movie);
     }            
     /*else
		{
			ctx.push(movie);

     }*/

		}
		
void print(){
    std::cout << "hello\n";
	} 

typedef int tt_has_serialize;
void serialize(Galois::Runtime::SerializeBuffer& s) const {
    gSerialize(s,g);
}
void deserialize(Galois::Runtime::DeSerializeBuffer& s) {
    gDeserialize(s,g);
} 

};
    void operator()(Graphp g) {
	DGraph::iterator ii = g->begin();
	std::advance(ii,num_movie_nodes);
	//std::advance(ii,2);
	
	//Graph::iterator jj = graph.begin();
 	//std::advance(jj, num_movie_nodes);

	Node& dg_movie = g->getData(*ii);
	//Node& g_movie = graph.getData(*jj);
	int movie_host0 = 0;
	int movie_host1 = 0;

	//Galois::for_each_local(g, Process(this,g), "Process");
/*	unsigned int blockSize = num_user_nodes/Galois::Runtime::NetworkInterface::Num;
	for(int i=0;i<Galois::Runtime::NetworkInterface::Num;i++) {
	    int blockNum = (Galois::Runtime::NetworkInterface::ID+i)%Galois::Runtime::NetworkInterface::Num;
	    unsigned int startRange = blockSize*blockNum;
	    unsigned int endRange = blockSize*(blockNum+1);
	    if(blockNum == Galois::Runtime::NetworkInterface::Num-1)	endRange = num_user_nodes+1;
	    startRange += num_movie_nodes+1;
	    endRange += num_movie_nodes+1;
	    cout<<"Iteration: "<<i<<endl;
*/
	    uint32_t startRange = 0;
	    uint32_t endRange = 0;
	    uint32_t t = Galois::Runtime::NetworkInterface::Num;
	    /* New for_each for blocking */
	    Galois::for_each<>(g->begin(), ii, t, num_user_nodes, Process(this,g,startRange,endRange), "SGD Process");
//	}
	//Galois::for_each(g->begin(), ii, verify_before(g), "Verifying");
    
        // Verification routine
  //      std::cout << "Running Verification after completion\n";
        //verify(g);
	std::cout << "number of nodes = "<<numNodes.reduce() << "\n";
    }

};


/*void fillNode(Node& node) {
    unsigned int seed = 42;
    std::default_random_engine eng(seed);
    std::uniform_real_distribution<double> random_lv_value(0,0.1);
   double *lv = new double[LATENT_VECTOR_SIZE];
    for(int i=0;i<LATENT_VECTOR_SIZE;i++) {
	lv[i] = random_lv_value(eng);
    }
    node.latent_vector = lv; 
}*/


static void recvHostIDMap_landing_pad(RecvBuffer& buf) {
    
    DGNode n;
    uint32_t hostID;
    gDeserialize(buf,hostID, n);
    slock.lock();
    HostIDMap[n] = hostID;
    slock.unlock();
}

/*
static void sendHostIDMap_landing_pad(RecvBuffer& buf) {
    
    DGNode n;
    uint32_t hostID;
    gDeserialize(buf, hostID, n);
    sendBuffer b;
    slock.lock();
    gserialize(b, Galois::Runtime::NetworkInterface::ID, n);
    slock.unlock();
    getSystemNetworkInterface().send(host, recvHostIDMap_landing_pad, b); 
}
*/


unsigned num_ns = 0;
struct create_nodes {
    Graphp g;
    SLock l;
    create_nodes() = default;
    create_nodes(Graphp _g, SLock _l) : g(_g),l(_l){}
    
    template<typename Context>
    void operator ()(const FileGNode& item, const Context& ctx) {
	Node node;
	DGNode n = g->createNode(node);  
	g->addNode(n);
    }
};

void giveDGraph(Graphp graph);
uint64_t countEdgeData=0;
static void create_remote_graph_edges(Graphp dgraph)
{
  printf ("creating all edges on HOST =>%u\n", Galois::Runtime::NetworkInterface::ID);
   unsigned count = 0;
   unsigned scount = 0;
   unsigned rcount = 0;

    unsigned cc = 0;
	
    auto dg_it = dgraph->begin();
    for(auto ii = fgraph.begin(); ii != fgraph.end(); ++ii) {
	mapping[*ii] = *dg_it;
	++dg_it;	
    } 
    idCount = 1;
    for(auto ii = fgraph.begin(); ii != fgraph.end(); ++ii) {
	FGraph::edge_iterator vv = fgraph.edge_begin(*ii);
	FGraph::edge_iterator ev = fgraph.edge_end(*ii);
	scount++;
	Node& n = dgraph->getData(mapping[*ii]);
	n.ID = idCount++;
//	cout << "n ID = "<< n.ID<<endl;
	for (FGraph::edge_iterator jj = vv; jj != ev; ++jj) {
	   // cout<<"Getting edges..\n";
	    Node& dst = dgraph->getData(mapping[fgraph.getEdgeDst(jj)]);
	    unsigned int dstID = dst.ID;
	    //cout<<"Value of ID = "<<dstID<<endl;
	    //cout<<"Value of edge data = "<<fgraph.getEdgeData<unsigned int>(jj)<<endl;
	    unsigned int edge_data = dstID*10+fgraph.getEdgeData<unsigned int>(jj);
	    //cout<<"Value of edge data again: "<<edge_data%10<<endl;
	    countEdgeData += edge_data;
	    dgraph->addEdge(mapping[*ii],mapping[fgraph.getEdgeDst(jj)], edge_data);
	    count++;
	    n.number_of_edges+=1;
	}
	if(n.number_of_edges > 0)
	    ++num_movie_nodes;
	else
	    ++num_user_nodes;
	NUM_RATINGS += n.number_of_edges;
  }
int movie_host0 = 0;
	int movie_host1 = 0;
	auto jj = dgraph->begin();
	std::advance(jj , num_movie_nodes);
	for(auto hh = dgraph->begin(); hh != jj; ++hh){
	    if(HostIDMap[*hh] == 0)
		++movie_host0;
	    if(HostIDMap[*hh] == 1)
		++movie_host1;
	} 

	std::cout<< "movie_host0 = " << movie_host0 << " moveie_host1 = " << movie_host1 <<std::endl;
}


static void create_dist_graph(Graphp dgraph, std::string inputFile) {
    SLock lk;
    uint64_t block, f, l;
    FGraph::iterator first, last;
    
    fgraph.structureFromFile(inputFile);
    auto graph_begin = fgraph.begin();
    auto graph_end = fgraph.end();

    auto  size = graph_end - graph_begin;;
    cout << "rest node size = "<<size<<endl;
    block = size / Galois::Runtime::NetworkInterface::Num;
    f = Galois::Runtime::NetworkInterface::ID * block;
    l = (Galois::Runtime::NetworkInterface::ID + 1) * block;
    first = graph_begin + (Galois::Runtime::NetworkInterface::ID * block);
    last  = graph_begin + ((Galois::Runtime::NetworkInterface::ID + 1) * block);
    if (Galois::Runtime::NetworkInterface::ID == (Galois::Runtime::NetworkInterface::Num-1)) last = graph_end;

    cout << "first - last" << (first - last)<< endl;
   // cout<<"Number of nodes = "<<std::distance(graph.begin(),graph.end())<<endl; 
   // if(Galois::Runtime::NetworkInterface::ID == 0) {   
//	std::cout << "number of node on host = " << Galois::Runtime::NetworkInterface::ID << " are: = " << last - first << "\n";
	printf ("host: %u creating rest of the nodes\n", Galois::Runtime::NetworkInterface::ID);
	Galois::for_each(first, last, create_nodes(dgraph,lk));
}
static void getDGraph_landing_pad(RecvBuffer& buf) {
    Graphp dgraph;
    gDeserialize(buf, dgraph);
   // printf("%d has received DistGraph..\n",Galois::Runtime::NetworkInterface::ID);
}
void giveDGraph(Graphp dgraph) {
    if(Galois::Runtime::NetworkInterface::Num > 1) {
    SendBuffer b;
    gSerialize(b, dgraph);
    getSystemNetworkInterface().broadcast(getDGraph_landing_pad, b);
//	printf("Handling receives...\n");
    getSystemNetworkInterface().handleReceives();
//	printf("Done Handling receives...\n");
    }
}
static void readInputGraph_landing_pad(RecvBuffer& buf) {
    Graphp dgraph;
    std::string inputFile;
    gDeserialize(buf, inputFile, dgraph);
    create_dist_graph(dgraph, inputFile);
 //   printf("1..Done creating dist graph..\n");
}


void readInputGraph(Graphp dgraph, std::string inputFile){
  //  std::cout<<"NetworkHostNum="<<Galois::Runtime::NetworkInterface::Num<<std::endl;
     if(Galois::Runtime::NetworkInterface::Num > 1) {
	SendBuffer b;
	gSerialize(b, inputFile, dgraph);
	getSystemNetworkInterface().broadcast(readInputGraph_landing_pad, b);
//	printf("Handling receives...\n");
	getSystemNetworkInterface().handleReceives();
//	printf("Done Handling receives...\n");
    } 
    
    create_dist_graph(dgraph,inputFile);
  //  printf("0..Done creating dist graph.. HOST --->%d\n", Galois::Runtime::NetworkInterface::ID);
	
}



void readGraph(Graphp dgraph, std::string inputFile) {
    readInputGraph(dgraph, inputFile); 
}

/*
void verify_(Graphp g) {
    auto ii = g->begin();
    std::advance(ii,num_movie_nodes);
    if(Galois::Runtime::NetworkInterface::ID == 0)	
	Galois::for_each(g->begin(),ii,verify_before(g),"Verifying");
    else
	Galois::for_each(g->begin(),ii,dummy_func2());
}
*/
int main(int argc, char** argv)
{	
	if(argc < 3)
	{
		std::cout << "Usage: <input binary gr file> <thread count>" << std::endl;
		return -1;
	}
	
	//std::cout<< "start reading and building Graph\n";
	std::string inputFile = argv[1];
	unsigned int threadCount = atoi(argv[2]);

	//how many threads Galois should use
	Galois::setActiveThreads(threadCount);
  //      graph.structureFromFile(inputFile);

//	num_movie_nodes =  initializeGraphData(graph);
//	std::cout << "num_movie_nodes = " << num_movie_nodes <<"\n";
//	cout<<"Number of ratings = "<<NUM_RATINGS<<endl;	
	Galois::StatManager statManager;
	Galois::Runtime::networkStart();
	
	Graphp dgraph = DGraph::allocate();

	Galois::StatTimer Tinitial("Initialization Time");
	Tinitial.start();
	readGraph(dgraph, inputFile);    
	Tinitial.stop();

//	if(Galois::Runtime::NetworkInterface::ID == 0) {	
	    std::cout<< "create_remote_graph_edges host--->" << Galois::Runtime::NetworkInterface::ID<<"\n";
	    create_remote_graph_edges(dgraph);
	    std::cout<< "Done reading and building Graph\n";
//	}
	//verify();
	cout<<"Verifying before SGD\n";
	verify(dgraph);
	std::cout << "calling sgd \n";
	Galois::StatTimer T("Sgd Time");

        NetworkInterface& net = getSystemNetworkInterface();

//	net.systemBarrier();
	T.start();
	sgd_algo()(dgraph);
	 T.stop();   
  //      net.systemBarrier();

/*	Galois::StatTimer T2("Verify Time");
	T2.start();
	cout<<"Verifying after SGD\n";
	verify(dgraph);
	T2.stop();
*/
	printf("NUMBER OF MOVIE NODES = %d\n", num_movie_nodes);
	cout<<"Sum of edge data values = "<<countEdgeData<<endl;
	Galois::Runtime::networkTerminate();
	return 0;

}
