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
 * Authors: Gurbinder Gill <gill@cs.utexas.edu>
 *	    Bharat Naik    <bharatpn@cs.utexas.edu>
 *
*/

#include <iostream>
#include <random>
#include <cmath>
#include <algorithm>
#include <utility>
#include <unordered_map>
#include <map>
#include <atomic>

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


#include <boost/iterator/transform_iterator.hpp>

typedef struct Node
{
  double* latent_vector; //latent vector to be learned
  unsigned int updates; //number of updates made to this node (only used by movie nodes)
  unsigned int edge_offset; //if a movie's update is interrupted, where to start when resuming
  unsigned int ID;  
    bool operator==(const Node& other) const
    {
	return (ID == other.ID);
    }
    bool operator<(const Node& other) const 
    {
	return (ID < other.ID);
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

//typedef galois::graphs::LC_Numa_Graph<Node, unsigned int> Graph;
//typedef Graph::GraphNode GNode;
/*typedef galois::graphs::FileGraph Graph;
typedef uint64_t GNode;
Graph File_graph;
*/

//Distributed Graph Nodes.
typedef galois::graphs::ThirdGraph<Node, unsigned int, galois::graphs::EdgeDirection::Out> DGraph;
typedef DGraph::NodeHandle DGNode;
typedef typename DGraph::pointer Graphp;


//typedef galois::graphs::FileGraph FGraph;
//typedef galois::graphs::FileGraph::GraphNode FileGNode;

typedef galois::graphs::LC_CSR_Graph<Node, unsigned int> Graph;
typedef Graph::GraphNode GNode;

Graph graph;

// TODO : replace maps with unordered_map
std::unordered_map<GNode,Node> lookup;
std::unordered_map<GNode,DGNode> mapping;
std::unordered_map<Node,DGNode> llookup;
std::unordered_map<Node,DGNode> rlookup;
std::set<Node> requested;

volatile unsigned prog_barrier = 0;
//std::atomic<unsigned> prog_barrier;
unsigned int num_movie_nodes = 0;

using namespace galois::Runtime;
typedef galois::runtime::LL::SimpleLock<true> SLock;
SLock slock;
SLock pblock;

unsigned int LATENT_VECTOR_SIZE = 20;
double LEARNING_RATE = 0.001;
double DECAY_RATE = 0.9;
double LAMBDA = 0.001;
unsigned int MAX_MOVIE_UPDATES = 1;
unsigned int NUM_RATINGS = 0;
static const double MINVAL = -1e+100;
static const double MAXVAL = 1e+100;


double vector_dot(const Node& movie_data, const Node& user_data) {
  const double* __restrict__ movie_latent = movie_data.latent_vector;
  const double* __restrict__ user_latent = user_data.latent_vector;

  double dp = 0.0;
  for (int i = 0; i < LATENT_VECTOR_SIZE; ++i)
    dp += user_latent[i] * movie_latent[i];
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
    getSystemLocalDirectory().makeProgress();
    getSystemRemoteDirectory().makeProgress();
    getSystemNetworkInterface().handleReceives();
  } while (prog_barrier != networkHostNum);

  prog_barrier = 0;
  printf("Left barrier..\n");
}


void verify(Graphp g){
    // Checking RMSE for verification
    typedef galois::GAccumulator<double> AccumDouble;
  //  unsigned int num_movie_nodes; 
    AccumDouble rms;
    //DGraph::iterator ii = g->begin() + num_movie_nodes;
    galois::do_all_local(g, [&g,&rms] (DGNode n) {
			for(auto ii = g->edge_begin(n); ii != g->edge_end(n); ++ii){
			
			    DGNode m = g->getEdgeDst(ii);
			    double pred = calcPrediction(g->getData(n), g->getData(m));
			    double rating = ii->getValue();

			    if(!std::isnormal(pred))
				std::cout << "Denormal Warning\n";
			    rms += ((pred - rating)*(pred - rating));
				
			}
			    
    });
    
    double total_rms = rms.reduce();
    double normalized_rms = sqrt(total_rms/NUM_RATINGS);
    std::cout << "RMSE Total: " << total_rms << " Normalized: " << normalized_rms << std::endl;

}

/* Operator */

struct sgd_algo {
   galois::GAccumulator<size_t> numNodes; 
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

     /* For checking if graph is correct  */
     /* DGraph::edge_iterator edge_begin = g->edge_begin(movie);
      DGraph::edge_iterator edge_end = g->edge_end(movie);
    
      Node& movie_data = g->getData(movie);
      std::cout << "Movie : " << movie_data.ID <<"\t";

      for(auto ii = edge_begin; ii != edge_end; ++ii)
      {
	    DGNode user = g->getEdgeDst(ii);
	    Node& user_data = g->getData(user);
	    unsigned int egde_data = ii->getValue();
	    std::cout << "User : " <<  user_data.ID <<"\t";
	    
      }	
	std::cout << "\n";
    */

     Node& movie_data = g->getData(movie);
     DGraph::edge_iterator edge_it = g->edge_begin(movie);
     DGraph::edge_iterator edge_end = g->edge_end(movie);
	
     std::advance(edge_it,  movie_data.edge_offset);
     DGNode user = g->getEdgeDst(edge_it);
     Node& user_data = g->getData(user);

     unsigned int edge_rating = edge_it->getValue();
     
    // Call the gradient routine
  //  doGradientUpdate(movie_data, user_data, edge_rating);
     
     ++edge_it;
     ++movie_data.edge_offset;

	   // This is the last user
     if(edge_it == g->edge_end(movie))// galois::MethodFlag::NONE))
     {
    //start back at the first edge again

	movie_data.edge_offset = 0;

	movie_data.updates++;
	if(movie_data.updates < MAX_MOVIE_UPDATES)
	    ctx.push(movie);
     }            
     else
     {
	  ctx.push(movie);
     }

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
	
	Graph::iterator jj = graph.begin();
 	std::advance(jj, num_movie_nodes);

	Node& dg_movie = g->getData(*ii);
	Node& g_movie = graph.getData(*jj);

	std::cout << "dg_movie = " << dg_movie.ID << "\n";
	std::cout << "g_movie = " << g_movie.ID << "\n";
	std::cout << "num movie nodes  = " << num_movie_nodes << "\n";
    
	


	//galois::for_each_local(g, Process(this,g), "Process");
	galois::for_each(g->begin(), ii, Process(this,g), "SGD Process");
    
        // Verification routine
        std::cout << "Running Verification after completion\n";
        //verify(g);
	

  //program_barrier();
	//std::cout << "number of nodes = "<<numNodes.reduce() << "\n";
    }

};
void fillNode(Node& node) {
    unsigned int seed = 42;
    std::default_random_engine eng(seed);
    std::uniform_real_distribution<double> random_lv_value(0,0.1);
   double *lv = new double[LATENT_VECTOR_SIZE];
    for(int i=0;i<LATENT_VECTOR_SIZE;i++) {
	lv[i] = random_lv_value(eng);
    }
    node.latent_vector = lv; 
}
struct create_nodes {
    Graphp g;
    SLock& l;
    create_nodes(Graphp _g, SLock& _l) : g(_g),l(_l){}
    
    void operator ()(GNode& item, galois::UserContext<GNode>& ctx) {
//	cout<<"In create nodes..\n";
	Node node =  graph.getData(item);
	//fillNode(node);
	if(node.ID == 1)
	    std::cout << "node ID = " << node.ID<<"Host =====>" << networkHostID<<std::endl;
	 DGNode n = g->createNode(node);  
	g->addNode(n);
	if(node.ID == 1)
	{
	    Node dNode = g->getData(n);
	     
	    std::cout << "node ID og dgraph = " << dNode.ID<<"Host =====>" << networkHostID<<std::endl;
	}
/*	l.lock();
	mapping[item] = n;
	llookup[lookup[item]] = n;
	l.unlock();
*/
    }

};

struct dummy_func {
    
    dummy_func(){} 
    void operator()(GNode& item, galois::UserContext<GNode>& ctx){}
};

unsigned int initializeGraphData(Graph& g)
{
        unsigned int seed = 42;
        std::default_random_engine eng(seed);
        std::uniform_real_distribution<double> random_lv_value(0, 0.1);

        unsigned int num_movie_nodes = 0;
	unsigned int num_user_nodes = 0;
	unsigned index = 0;
	unsigned int numRatings = 0;
     //  for all movie and user nodes in the graph
	    for (Graph::iterator i = g.begin(), end = g.end(); i != end; ++i) {
		Graph::GraphNode gnode = *i;
		Node& data = g.getData(gnode);
		
		data.ID = index;
		++index; 
		data.updates = 0;
    
		 //fill latent vectors with random values
                 double* lv = new double[LATENT_VECTOR_SIZE];
		 for(int i = 0; i < LATENT_VECTOR_SIZE; i++)
                 {
                     lv[i] = random_lv_value(eng);
                 }
                 data.latent_vector = lv;
                                                                                                                                                                                                                        unsigned int num_edges = g.edge_end(gnode) - g.edge_begin(gnode);
	    //std::cout << "num edges = " << num_edges << "\n";
	    numRatings += num_edges;
	    if(num_edges > 0)
                 ++num_movie_nodes;
	    else 
		++num_user_nodes;
	    
	//     data.edge_offset = 0;
	} 
	
	NUM_RATINGS = numRatings;
        return num_movie_nodes;
}



static void recvRemoteNode_landing_pad(RecvBuffer& buf) {
  DGNode n;
  //unsigned num;
  Node node;
  uint32_t host;
  gDeserialize(buf,host,node,n);
  slock.lock();
  rlookup[node] = n;
  slock.unlock();
}

static void getRemoteNode_landing_pad(RecvBuffer& buf) {
 // unsigned num;
  Node node;
  uint32_t host;
  gDeserialize(buf,host,node);
  SendBuffer b;
  slock.lock();
  gSerialize(b, networkHostID, node, llookup[node]);
  slock.unlock();
  getSystemNetworkInterface().send(host,recvRemoteNode_landing_pad,b);
}

static void create_remote_graph_edges(Graphp dgraph)
{
  printf ("host: %u creating all edges on HOST =>\n", networkHostID);
   unsigned count = 0;
   unsigned scount = 0;
   unsigned rcount = 0;

    unsigned cc = 0;

    /*auto start = graph.begin();
    std::advance(start,0);
    DGNode d_start = mapping[*start];
    Node d_node = dgraph->getData(d_start);
    
    auto d_begin = dgraph->begin();
    Node d_start_node = dgraph->getData(*d_begin);
    std::cout << "---------------d graph mapping at = >" <<d_node.ID <<std::endl;
    std::cout << "---------------d graph starting at = >" <<d_start_node.ID <<std::endl;
    */
   //rlookup.clear();
   //assert(!rlookup.size()); 
    auto dg_it = dgraph->begin();
    for(auto ii = graph.begin(); ii != graph.end(); ++ii) {
	mapping[*ii] = *dg_it;
	++dg_it;	
    } 
    for(auto ii = graph.begin(); ii != graph.end(); ++ii) {
	Graph::edge_iterator vv = graph.edge_begin(*ii);
	Graph::edge_iterator ev = graph.edge_end(*ii);
	scount++;
	for (Graph::edge_iterator jj = vv; jj != ev; ++jj) {
	    //Node& node = lookup[graph.getEdgeDst(jj)];
	    unsigned int edge_data = graph.getEdgeData(jj);
      
	    dgraph->addEdge(mapping[*ii],mapping[graph.getEdgeDst(jj)], edge_data);
	    count++;
	}
  }
  std::cout <<"host="<<networkHostID<<"count = " << count <<"\n";
    
  printf ("host: %u nodes %u and edges %u remote edges %u\n", networkHostID, scount, count, rcount);
  printf ("host: %u done creating local edges\n", networkHostID);
  
}


static void create_dist_graph(Graphp dgraph, std::string inputFile) {
    SLock lk;
    prog_barrier = 0;
    uint64_t block, f, l;
    Graph::iterator first, last;
    std::cout<<"Done making graph HOST = " << networkHostID <<"\n"; 
    std::cout<<"Number of movie nodes="<<num_movie_nodes<<std::endl; 
    unsigned size = 0;
    cout<<"Number of nodes = "<<std::distance(graph.begin(),graph.end())<<endl; 
   // std::cout << "host = "<< networkHostID << " f  = "<< f << " l = " << l <<"\n";

  // create the nodes
    if(networkHostID == 0) {   
    printf ("host: %u creating nodes\n", networkHostID);
   
    galois::for_each(graph.begin(),graph.end(),create_nodes(dgraph,lk));
   //printf ("%lu nodes in %u host with block size %lu\n", mapping.size(), networkHostID, block);
   // create the local edges
}
else {
    
    galois::for_each(graph.begin(),graph.end(),dummy_func());
}
}

static void readInputGraph_landing_pad(RecvBuffer& buf) {
    Graphp dgraph;
    std::string inputFile;
    gDeserialize(buf, inputFile, dgraph);
    create_dist_graph(dgraph, inputFile);
    printf("1..Done creating dist graph..\n");
}


void readInputGraph(Graphp dgraph, std::string inputFile){
    std::cout<<"NetworkHostNum="<<networkHostNum<<std::endl;
     if(networkHostNum > 1) {
	SendBuffer b;
	gSerialize(b, inputFile, dgraph);
	getSystemNetworkInterface().broadcast(readInputGraph_landing_pad, b);
	printf("Handling receives...\n");
	getSystemNetworkInterface().handleReceives();
	printf("Done Handling receives...\n");
    } 
    
    create_dist_graph(dgraph,inputFile);
    printf("0..Done creating dist graph.. HOST --->%d\n", networkHostID);
	
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
	
	std::cout<< "start reading and building Graph\n";
	//read command line parameters
	//const char* inputFile = argv[1];
	std::string inputFile = argv[1];
	unsigned int threadCount = atoi(argv[2]);

	//how many threads Galois should use
	galois::setActiveThreads(threadCount);
	//FGraph fg;
	//fg.structureFromFile(inputFile);
	//std::cout<<"Reached here..\n";
        //graph.structureFromGraph(fg);
	//std::cout<<"Reached here too..\n";

	//prints out the number of conflicts at the end of the program
	std::cout << "structureFromFile\n";	
        graph.structureFromFile(inputFile);

	num_movie_nodes =  initializeGraphData(graph);
	std::cout << "num_movie_nodes = " << num_movie_nodes <<"\n";	

	galois::StatManager statManager;
	galois::runtime::networkStart();
	
	Graphp dgraph = DGraph::allocate();

	galois::StatTimer Tinitial("Initialization Time");
	Tinitial.start();
	readGraph(dgraph, inputFile);    
	Tinitial.stop();

	if(networkHostID == 0) {	
	std::cout<< "create_remote_graph_edges host--->" << networkHostID<<"\n";
        create_remote_graph_edges(dgraph);
	std::cout<< "Done reading and building Graph\n";
	}
	std::cout<< "Running Verification before \n";
//	verify(dgraph);

    
	std::cout << "num_movie_nodes = " << num_movie_nodes << "\n";
	std::cout << "calling sgd \n";
	//program_barrier();
	galois::StatTimer T("Sgd Time");
	T.start();
	sgd_algo()(dgraph);
	 T.stop();   


//allocate local computation graph
/*	FGraph f;
	f.structureFromFile(inputFile);
	LCGraph localGraph;
	localGraph.allocateFrom(f);
	localGraph.constructFrom(f, 0, 1);
	//read structure of graph & edge weights; nodes not initialized
	//galois::graphs::readGraph(localGraph, inputFile);
	//g_ptr = &g;

	//typedef galois::graphs::FileGraph::GraphNode FileGNode;
	for(auto ii = localGraph.begin(); ii != localGraph.end(); ++ii) {
		Node& data = localGraph.getData(*ii);
		std::cout << data.updates <<"\n";

	}

	//fill each node's id & initialize the latent vectors
	unsigned int num_movie_nodes = initializeGraphData(localGraph);

	Graphp g = Graph::allocate();

	//galois::for_each<>(boost::counting_iterator<int>(0), boost::counting_iterator<int>(100), op(g));

	//galois::for_each<>(f.begin(), f.end(), AddNodes(g,f));

	std::cout << "Done making graph\n";

	
	galois::StatTimer timer;
	timer.start();

	//do the SGD computation in parallel
	//the initial worklist contains all the movie nodes
	//the movie nodes are located at the top num_movie_nodes nodes in the graph
	//the worklist is a priority queue ordered by the number of updates done to a movie
	//the projCount functor provides the priority function on each node
	//Graphp::iterator ii = g.begin();
	//std::advance(ii,num_movie_nodes); //advance moves passed in iterator
    galois::for_each(g.begin(), ii, sgd(g),
                         galois::wl<galois::WorkList::OrderedByIntegerMetric
                         <projCount, galois::WorkList::dChunkedLIFO<32>>>());

	timer.stop();
	
	std::cout << "SUMMARY Movies " << num_movie_nodes <<
		" Users " << g->size() - num_movie_nodes<<
		" Threads " << threadCount << 
		" Time " << timer.get()/1000.0 << std::endl;

*/
	printf("Reached here, before terminate..\n");
	galois::runtime::networkTerminate();
	printf("Reached here, after terminate..\n");
	return 0;
}
