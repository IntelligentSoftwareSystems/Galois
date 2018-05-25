/**
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of XYZ License (a copy is located in
 * LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
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

#include "galois/Galois.h"
#include "galois/Graph/Graph.h"
#include "galois/Graph/LCGraph.h"
#include "galois/ParallelSTL/ParallelSTL.h"
#include "llvm//Support/CommandLine.h"
#include "Lonestar/BoilerPlate.h"
#include "galois/runtime/Network.h"
#include "galois/Timer.h"
#include "galois/Timer.h"
#include "galois/Graph/FileGraph.h"
//Distributed Galois
#include "galois/graphs/Graph3.h"
#include "galois/runtime/DistSupport.h"


#include <boost/iterator/transform_iterator.hpp>
#define LATENT_VECTOR_SIZE 2
typedef int EdgeData;
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
    void serialize(galois::runtime::SerializeBuffer& s) const {
	gSerialize(s,latent_vector, updates, edge_offset, ID, number_of_edges);
    }
    void deserialize(galois::runtime::DeSerializeBuffer& s) {
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

//Distributed Graph Nodes.
typedef galois::graphs::ThirdGraph<Node, uint32_t, galois::graphs::EdgeDirection::Out> DGraph;
typedef DGraph::NodeHandle DGNode;
typedef typename DGraph::pointer Graphp;


typedef galois::graphs::FileGraph FGraph;
typedef galois::graphs::FileGraph::GraphNode FileGNode;
FGraph fgraph;

//Graph graph;
std::unordered_map<FileGNode,DGNode> mapping;
// DGNode ==> hostID mapping
std::map<DGNode, uint64_t> HostIDMap;

using accumulator = galois::GAccumulator<int64_t>;

//Processed movie nodes:
uint64_t Processed_movie_nodes = 0;

volatile unsigned prog_barrier = 0;
//std::atomic<unsigned> prog_barrier;
//unsigned int num_movie_nodes = 0;
unsigned int num_movie_nodes = 0;

using namespace galois::runtime;
typedef galois::runtime::LL::SimpleLock<true> SLock;
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

inline void doGradientUpdate(Node& movie_data, Node& user_data, unsigned int edge_rating)
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


/*
inline void doGradientUpdate(Node& movie_data, Node& user_data, unsigned int edge_rating)
{
        double step_size = LEARNING_RATE * 1.5 / (1.0 + DECAY_RATE * pow(movie_data.updates + 1, 1.5));
        double* __restrict__ movie_latent = movie_data.latent_vector;
        double* __restrict__ user_latent = user_data.latent_vector;
	
	//calculate error
        double cur_error = - edge_rating;
        for(unsigned int i = 0; i < LATENT_VECTOR_SIZE; i++)
        {
                cur_error += user_latent[i] * movie_latent[i];
        }

	//This is a gradient step
        for(unsigned int i = 0; i < LATENT_VECTOR_SIZE; i++)
        {
                double prev_movie_val = movie_latent[i];

                movie_latent[i] -= step_size * (cur_error * user_latent[i] + LAMBDA * prev_movie_val);
                user_latent[i] -= step_size * (cur_error * prev_movie_val + LAMBDA * user_latent[i]);
        }
}

*/

static void progBarrier_landing_pad(RecvBuffer& buf) {
   gDeserialize(buf,prog_barrier);
    ++prog_barrier;
}

static void program_barrier() {
  SendBuffer b;
  gSerialize(b, prog_barrier);
  getSystemNetworkInterface().broadcast(progBarrier_landing_pad, b);

//unsigned old_val = prog_barrier;
 //unsigned new_val =  ++prog_barrier;
    prog_barrier++;
//prog_barrier.compare_exchange_strong(old_val,new_val);
    printf("Entering barrier..%d\n", prog_barrier);
  do {
    //std::cout << "inside do loop\n";
    getSystemDirectory().makeProgress();
   // getSystemRemoteDirectory().makeProgress();
    getSystemNetworkInterface().handleReceives();
  } while (prog_barrier != networkHostNum);

  prog_barrier = 0;
  printf("Left barrier..\n");
}

galois::GAccumulator<double> RMS;
galois::GAccumulator<unsigned> count_data;
void verify(Graphp g){
    typedef galois::GAccumulator<double> AccumDouble;
    AccumDouble rms;
    cout<<"Host:"<<networkHostID<<" is verifying after SGD..\n";
    //galois::do_all(g, [&g,&rms] (DGNode n) {
	auto ei = g->begin();
	std::advance(ei,num_movie_nodes);
	unsigned int count=0;
	for(auto ni = g->begin(); ni != ei; ++ni) {
			for(auto ii = g->edge_begin(*ni); ii != g->edge_end(*ni); ++ii){
			
			    DGNode m = g->getEdgeDst(ii);
			    double pred = calcPrediction(g->getData(*ni), g->getData(m));
			    double rating = ii->getValue();
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
galois::GAccumulator<size_t> numNodes; 
unsigned count_done=0;
struct sgd_algo {
   //unsigned int num_movie_nodes;

struct Process : public galois::runtime::Lockable {
    Graphp g;
    sgd_algo* self;
    Process(){ }
    // sgd(Graphp _g) : g(_g) {}
    Process(sgd_algo* s, Graphp _g) : g(_g), self(s) { }
    //void operator()(const DGNode& n, galois::UserContext<DGNode>&) {(*this)(n);} 
    void operator()(const DGNode& movie, galois::UserContext<DGNode>& ctx)
    {
     Node& movie_data = g->getData(movie); 
    //printNode(movie_data);

     DGraph::edge_iterator edge_it = g->edge_begin(movie);
     DGraph::edge_iterator edge_end = g->edge_end(movie);
/**********************************************************
* Process all the edges of a movie node in 
* in one go.
* ********************************************************/ 	
    for(auto ii = edge_it; ii != edge_end; ++ii) {
	DGNode user = g->getEdgeDst(edge_it);
	Node& user_data = g->getData(user);
	
	unsigned int edge_rating = edge_it->getValue();
    
	doGradientUpdate(movie_data, user_data, edge_rating);
	//++movie_data.edge_offset;
    }
    
    numNodes += 1;
    ++Processed_movie_nodes;

}


typedef int tt_has_serialize;
void serialize(galois::runtime::SerializeBuffer& s) const {
    gSerialize(s,g);
}
void deserialize(galois::runtime::DeSerializeBuffer& s) {
    gDeserialize(s,g);
} 

};
    void operator()(Graphp g) {
	DGraph::iterator ii = g->begin();
	std::advance(ii,num_movie_nodes);
	Node& dg_movie = g->getData(*ii);
	//galois::for_each(g, Process(this,g), "Process");
	galois::for_each(g->begin(), ii, Process(this,g), "SGD Process");
	//galois::for_each(g->begin(), ii, verify_before(g), "Verifying");
   	std::cout << "number of nodes = "<<numNodes.reduce() << "\n";
    }

};

static void recvHostIDMap_landing_pad(RecvBuffer& buf) {
    
    DGNode n;
    uint32_t hostID;
    gDeserialize(buf,hostID, n);
    slock.lock();
    HostIDMap[n] = hostID;
    slock.unlock();
}


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
/*
static void recvRemoteNode_landing_pad(RecvBuffer& buf) {
  DGNode n;
  Node node;
  uint32_t host;
  gDeserialize(buf,host,node,n);
  slock.lock();
  rlookup[node] = n;
  slock.unlock();
}

static void getRemoteNode_landing_pad(RecvBuffer& buf) {
  Node node;
  uint32_t host;
  gDeserialize(buf,host,node);
  SendBuffer b;
  slock.lock();
  gSerialize(b, networkHostID, node, llookup[node]);
  slock.unlock();
  getSystemNetworkInterface().send(host,recvRemoteNode_landing_pad,b);
}

*/


static void create_remote_graph_edges(Graphp dgraph)
{
  printf ("creating all edges on HOST =>%u\n", networkHostID);
   unsigned count = 0;
   unsigned scount = 0;
   unsigned rcount = 0;

    unsigned cc = 0;
	
    auto dg_it = dgraph->begin();
    cout<<"Started mapping.."<<endl;
    for(auto ii = fgraph.begin(); ii != fgraph.end(); ++ii) {
	mapping[*ii] = *dg_it;
	++dg_it;	
    }
    cout<<"Done with mapping"<<endl; 
    for(auto ii = fgraph.begin(); ii != fgraph.end(); ++ii) {
	FGraph::edge_iterator vv = fgraph.edge_begin(*ii);
	FGraph::edge_iterator ev = fgraph.edge_end(*ii);
	scount++;
	Node& n = dgraph->getData(mapping[*ii]);
//	cout << "n ID = "<< n.ID<<endl;
	for (FGraph::edge_iterator jj = vv; jj != ev; ++jj) {
	   // cout<<"Getting edges..\n";
	    unsigned int edge_data = fgraph.getEdgeData<unsigned int>(jj);
	    dgraph->addEdge(mapping[*ii],mapping[fgraph.getEdgeDst(jj)], edge_data);
	    count++;
	    n.number_of_edges+=1;
	}
	if(n.number_of_edges > 0)
	    ++num_movie_nodes;

	NUM_RATINGS += n.number_of_edges;
  }

  /*  auto uu = dgraph->begin();
    std::advance(uu, num_movie_nodes+2);

    for(auto g = dgraph->begin(); g != uu; ++g)
    {
	cout << "n edges = " <<dgraph->getData(*g).number_of_edges<<endl; 
    }
*/
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


 // std::cout <<"host="<<networkHostID<<" count = " << count <<"\n";
    
 // printf ("host: %u nodes %u and edges %u remote edges %u\n", networkHostID, scount, count, rcount);
//  printf ("host: %u done creating local edges\n", networkHostID);
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
    block = size / networkHostNum;
    f = networkHostID * block;
    l = (networkHostID + 1) * block;
    first = graph_begin + (networkHostID * block);
    last  = graph_begin + ((networkHostID + 1) * block);
    if (networkHostID == (networkHostNum-1)) last = graph_end;

    cout << "first - last" << (first - last)<< endl;
    printf ("host: %u creating rest of the nodes\n", networkHostID);
    galois::for_each(first, last, create_nodes(dgraph,lk));
}
static void getDGraph_landing_pad(RecvBuffer& buf) {
    Graphp dgraph;
    gDeserialize(buf, dgraph);
   // printf("%d has received DistGraph..\n",networkHostID);
}
void giveDGraph(Graphp dgraph) {
    if(networkHostNum > 1) {
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
  //  std::cout<<"NetworkHostNum="<<networkHostNum<<std::endl;
     if(networkHostNum > 1) {
	SendBuffer b;
	gSerialize(b, inputFile, dgraph);
	getSystemNetworkInterface().broadcast(readInputGraph_landing_pad, b);
//	printf("Handling receives...\n");
	getSystemNetworkInterface().handleReceives();
//	printf("Done Handling receives...\n");
    } 
    
    create_dist_graph(dgraph,inputFile);
  //  printf("0..Done creating dist graph.. HOST --->%d\n", networkHostID);
	
}



void readGraph(Graphp dgraph, std::string inputFile) {
    readInputGraph(dgraph, inputFile); 
}

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
	galois::setActiveThreads(threadCount);
  //      graph.structureFromFile(inputFile);

//	num_movie_nodes =  initializeGraphData(graph);
//	std::cout << "num_movie_nodes = " << num_movie_nodes <<"\n";
//	cout<<"Number of ratings = "<<NUM_RATINGS<<endl;	
	galois::StatManager statManager;
	galois::runtime::networkStart();
	
	Graphp dgraph = DGraph::allocate();

	galois::StatTimer Tinitial("Initialization Time");
	Tinitial.start();
	readGraph(dgraph, inputFile);    
	Tinitial.stop();

//	if(networkHostID == 0) {	
	    std::cout<< "create_remote_graph_edges host--->" << networkHostID<<"\n";
	    create_remote_graph_edges(dgraph);
	    std::cout<< "Done reading and building Graph\n";
//	}
	//verify();
	cout<<"Verifying before SGD\n";
	verify(dgraph);
	std::cout << "calling sgd \n";
	galois::StatTimer T("Sgd Time");
	T.start();
	sgd_algo()(dgraph);
	 T.stop();   
	galois::StatTimer T2("Verify Time");
	T2.start();
	cout<<"Verifying after SGD\n";
	verify(dgraph);
	T2.stop();
	printf("NUMBER OF MOVIE NODES = %d\n", num_movie_nodes);
	galois::runtime::networkTerminate();
	return 0;

}
