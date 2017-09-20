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

typedef galois::graphs::LC_Numa_Graph<Node, unsigned int> Graph;
typedef Graph::GraphNode GNode;
//typedef galois::graphs::FileGraph Graph;
//typedef Graph::GraphNode GNode;
/*typedef galois::graphs::LC_PartitionedInlineEdge_Graph<Node, uint32_t>
    ::template with_out_of_line_lockable<true>
    ::template with_compressed_node_ptr<true>
    ::template with_numa_alloc<true>
    Graph;
  typedef typename Graph::GraphNode GNode;*/
Graph graph;
unsigned int track=0;
using std::cout;
using std::endl;
//Distributed Graph Nodes.
typedef galois::graphs::ThirdGraph<Node, unsigned int, galois::graphs::EdgeDirection::Out> DGraph;
typedef DGraph::NodeHandle DGNode;
typedef typename DGraph::pointer Graphp;

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

unsigned int LATENT_VECTOR_SIZE = 3;
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

    if(!movie_latent) std::cout<<"Bad movie..\n";
    if(!user_latent) std::cout<<"Bad user..\n";
  double dp = 0.0;
    //std::cout<<"Performing dp inside dp..\n";
    //std::cout<<"Movie data ID = "<<movie_data.ID<<" and user data ID = "<<user_data.ID<<std::endl;
    std::cout<<"Movie: "<<movie_latent[0]<<std::endl;
    std::cout<<"User: "<<user_latent[0]<<std::endl;
  for (int i = 0; i < LATENT_VECTOR_SIZE; ++i) {
    //std::cout<<i<<std::endl;
    dp += user_latent[i] * movie_latent[i];
    //std::cout<<i<<std::endl;
  }
  //std::cout<<"Done Performing dp inside dp..\n";
  assert(std::isnormal(dp));
  return dp;
}

double calcPrediction (const Node& movie_data, const Node& user_data) {
  double pred = vector_dot(movie_data, user_data);
  pred = std::min (MAXVAL, pred);
  pred = std::max (MINVAL, pred);
  return pred;
}


void printNode(const Node& t) {
    cout<<"ID: "<<t.ID<<endl;
    cout<<"Edge_offset: "<<t.edge_offset<<endl;
    cout<<"Updates: "<<t.updates<<endl;
    for(int i=0;i<LATENT_VECTOR_SIZE;i++) {
	cout<<" "<<t.latent_vector[i]<<endl;
    }
}


void doGradientUpdate(Node& movie_data, Node& user_data, unsigned int edge_rating)
{
    //std::cout<<"Retrieving movie data latent vector..\n";
  double* __restrict__ movie_latent = movie_data.latent_vector;
    //std::cout<<"Done Retrieving movie data latent vector..\n";
    //std::cout<<"Retrieving movie data updates..\n";
        double step_size = LEARNING_RATE * 1.5 / (1.0 + DECAY_RATE * pow(movie_data.updates + 1, 1.5));
    //std::cout<<"Done Retrieving movie data updates..\n";
    //std::cout<<"Retrieving user data latent vector..\n";
  double* __restrict__ user_latent = user_data.latent_vector;
    //std::cout<<"Done Retrieving user data latent vector..\n";
    //std::cout<<"Performing dp..\n";
  double cur_error = edge_rating - vector_dot(movie_data, user_data);
    //std::cout<<"Done Performing dp..\n";

    //std::cout<<"Updating vectors..\n";
  for(unsigned int i = 0; i < LATENT_VECTOR_SIZE; i++)
    {
      double prev_movie_val = movie_latent[i];
      double prev_user_val = user_latent[i];
      movie_latent[i] += step_size * (cur_error * prev_user_val  - LAMBDA * prev_movie_val);
      user_latent[i]  += step_size * (cur_error * prev_movie_val - LAMBDA * prev_user_val);
    }
    //std::cout<<"Updated vectors..\n";
}

/*
inline void doGradientUpdate(Node& movie_data, Node& user_data, unsigned int edge_rating)
{
        double step_size = LEARNING_RATE * 1.5 / (1.0 + DECAY_RATE * pow(movie_data.updates + 1, 1.5));
        double* __restrict__ movie_latent = movie_data.latent_vector;
        double* __restrict__ user_latent = user_data.latent_vector;
	
	//calculate error
      //  double cur_error = - edge_rating;
 //calculate error
        double cur_error = edge_rating - vector_dot(movie_data, user_data);
 
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

void shared_verify(){
    typedef galois::GAccumulator<double> AccumDouble;
    AccumDouble rms;
               for(Graph::iterator iii = graph.begin(); iii != graph.end(); ++iii) { 
			for(Graph::edge_iterator ii = graph.edge_begin(*iii); ii != graph.edge_end(*iii); ++ii){
			
			    GNode m = graph.getEdgeDst(ii);
			    double pred = calcPrediction(graph.getData(*iii), graph.getData(m));
			    double rating = graph.getEdgeData(ii);

			    if(!std::isnormal(pred))
				std::cout << "Denormal Warning\n";
			    rms += ((pred - rating)*(pred - rating));
				
			}
			    
    }
    
    double total_rms = rms.reduce();
    double normalized_rms = sqrt(total_rms/NUM_RATINGS);
    std::cout << "RMSE Total: " << total_rms << " Normalized: " << normalized_rms << std::endl;

}





struct shared_sgd_algo {
   galois::GAccumulator<size_t> numNodes; 
   //unsigned int num_movie_nodes;

struct Process : public galois::runtime::Lockable {
    shared_sgd_algo* self;
    //Process(){ }
    // sgd(Graphp _g) : g(_g) {}
    Process(shared_sgd_algo* s) : self(s) { }
    //void operator()(const DGNode& n, galois::UserContext<DGNode>&) {(*this)(n);} 
    void operator()(const GNode& movie, galois::UserContext<GNode>& ctx)
    {

     /* For checking if graph is correct  */
    /*  DGraph::edge_iterator edge_begin = g->edge_begin(movie);
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
    std::cout<<"reached here..\n";
     Node& movie_data = graph.getData(movie);
     Graph::edge_iterator edge_it = graph.edge_begin(movie);
     Graph::edge_iterator edge_end = graph.edge_end(movie);
	
     std::advance(edge_it,  movie_data.edge_offset);
     GNode user = graph.getEdgeDst(edge_it);
     Node& user_data = graph.getData(user);

     //unsigned int edge_rating = edge_it->getValue();
     unsigned int edge_rating = graph.getEdgeData(edge_it);//edge_it->getValue();
    if(edge_rating >  0)
   std::cout << "edge rating = " << edge_rating <<"\n";
 
    // Call the gradient routine
    //doGradientUpdate(movie_data, user_data, edge_rating);
     ++edge_it;
     ++movie_data.edge_offset;

	   // This is the last user
     if(edge_it == graph.edge_end(movie))// galois::MethodFlag::NONE))
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


};
    void operator()() {
	Graph::iterator ii = graph.begin();
	std::cout << "number of movie nodes inside sgd =============>" << num_movie_nodes <<"\n";
	std::advance(ii,num_movie_nodes);
	
//	Graph::iterator jj = graph.begin();
//	std::advance(jj, num_movie_nodes);

/*	Node& dg_movie = g->getData(*ii);
	Node& g_movie = graph.getData(*jj);

	std::cout << "dg_movie = " << dg_movie.ID << "\n";
	std::cout << "g_movie = " << g_movie.ID << "\n";
	std::cout << "num movie nodes  = " << num_movie_nodes << "\n";
    
	DGNode dg_begin= mapping[*jj];
	Node lookup_node = lookup[*jj];
	DGNode lookup_dgnode = llookup[lookup_node];
	if(dg_begin == lookup_dgnode)
	    std::cout << "matchde\n";
//	std::cout << "mapping"<< mapping[*jj] " << dg_movie.ID << "\n";
	
*/

	//galois::for_each_local(g, Process(this,g), "Process");
	std::cout<<std::distance(graph.begin(),ii)<<std::endl;
	galois::for_each(graph.begin(), ii, Process(this), "SGD Process");
    
        // Verification routine
        std::cout << "Running Verification after completion\n";
        //shared_verify();
	

  //program_barrier();
	//std::cout << "number of nodes = "<<numNodes.reduce() << "\n";
    }

};








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
    /*  DGraph::edge_iterator edge_begin = g->edge_begin(movie);
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
    //std::cout<<"Reached here...\n";
     Node& movie_data = g->getData(movie);
     DGraph::edge_iterator edge_it = g->edge_begin(movie);
     DGraph::edge_iterator edge_end = g->edge_end(movie);
    //std::cout<<std::distance(edge_it,edge_end)<<std::endl;
     //if(edge_it == edge_end)
//	    std::cout<<"Mar gaye..\n";
    //if(movie_data.edge_offset > 0)	
//	    std::cout<<movie_data.edge_offset<<" Mar gaye..\n";
     std::advance(edge_it,  movie_data.edge_offset);
     DGNode user = g->getEdgeDst(edge_it);
     Node& user_data = g->getData(user);

     unsigned int edge_rating = edge_it->getValue();
 /*   if(edge_rating >  0)
    std::cout << "edge rating = " << edge_rating <<"\n";
 */
    // Call the gradient routine
    //std::cout<<"Performing update..\n";
    doGradientUpdate(movie_data, user_data, edge_rating);
    //std::cout<<"Done performing update..\n";
    //std::cout<<"Printing data..\n";
    //std::cout<<movie_data.updates<<"\n"; 
    //std::cout<<"Done printing data..\n"; 
     
     ++edge_it;
     ++movie_data.edge_offset;
	   // This is the last user
     if(edge_it == g->edge_end(movie))// galois::MethodFlag::NONE))
     {
    //start back at the first edge again

	//std::cout<<"Reached here...\n";
	movie_data.edge_offset = 0;

	movie_data.updates++;
	if(movie_data.updates < MAX_MOVIE_UPDATES)
	    ctx.push(movie);
     }            
     else
     {
	//std::cout<<"Not Reached here...\n";
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
	std::cout << "number of movie nodes inside sgd =============>" << num_movie_nodes <<"\n";
	std::advance(ii,num_movie_nodes);
	
//	Graph::iterator jj = graph.begin();
//	std::advance(jj, num_movie_nodes);

/*	Node& dg_movie = g->getData(*ii);
	Node& g_movie = graph.getData(*jj);

	std::cout << "dg_movie = " << dg_movie.ID << "\n";
	std::cout << "g_movie = " << g_movie.ID << "\n";
	std::cout << "num movie nodes  = " << num_movie_nodes << "\n";
    
	DGNode dg_begin= mapping[*jj];
	Node lookup_node = lookup[*jj];
	DGNode lookup_dgnode = llookup[lookup_node];
	if(dg_begin == lookup_dgnode)
	    std::cout << "matchde\n";
//	std::cout << "mapping"<< mapping[*jj] " << dg_movie.ID << "\n";
	
*/

	//galois::for_each_local(g, Process(this,g), "Process");
	galois::for_each(g->begin(), ii, Process(this,g), "SGD Process");
    
        // Verification routine
        std::cout << "Running Verification after completion\n";
        //verify(g);
	

  //program_barrier();
	//std::cout << "number of nodes = "<<numNodes.reduce() << "\n";
    }

};

struct create_nodes {
    Graphp g;
    SLock& l;
    create_nodes(Graphp _g, SLock& _l) : g(_g),l(_l){}
    
    void operator ()(GNode& item, galois::UserContext<GNode>& ctx) {
	Node& node = graph.getData(item, galois::MethodFlag::NONE);
	DGNode n = g->createNode(node);  
	g->addNode(n);
/*	l.lock();
	mapping[item] = n;
	llookup[lookup[item]] = n;
	l.unlock();
*/
    }

};


unsigned int initializeGraphData(Graph& g)
{

std::cout << "inside Initialize  On Host = " << networkHostID <<"\n";
        unsigned int seed = 42;
        std::default_random_engine eng(seed);
        std::uniform_real_distribution<double> random_lv_value(0, 0.1);

        unsigned int num_movie_nodes = 0;
	unsigned int num_user_nodes = 0;
	unsigned index = 0;
	unsigned int numRatings = 0;
     //  for all movie and user nodes in the graph
	    for (auto i = g.begin(), end = g.end(); i != end; ++i) {
		auto dgnode = *i;
		Node& data = g.getData(dgnode);
		
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
              unsigned int num_edges = g.edge_end(dgnode) - g.edge_begin(dgnode);
	    //std::cout << "num edges = " << num_edges << "\n";
	    numRatings += num_edges;
	    if(num_edges > 0)
                 ++num_movie_nodes;
	    else 
		++num_user_nodes;
	    
	     data.edge_offset = 0;
	}
	//for(auto i = g.begin(), end = g.end(); i != end; ++i)
	  //  printNode(g.getData(*i)); 
	
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




static void create_dist_graph(Graphp dgraph, std::string inputFile) {
    SLock lk;
    prog_barrier = 0;
    uint64_t block, f, l;
	graph.structureFromFile(inputFile);
    Graph::iterator first, last;
std::cout << "inside creat_dist On Host = " << networkHostID <<"\n";
std::cout << "READ creat_dist On Host = " << networkHostID <<"\n";
	num_movie_nodes =  initializeGraphData(graph);
   std::cout<<"Number of movie nodes="<<num_movie_nodes<<std::endl; 
    unsigned size = 0; 
    for(auto ii = graph.begin(); ii!=graph.end();++ii) {
	lookup[*ii] = graph.getData(*ii);
	++size;
    }
    std::cout<<"Size="<<size<<std::endl;
    
  block = size / networkHostNum;
  f = networkHostID * block;
  l = (networkHostID + 1) * block;
  first = graph.begin() + (networkHostID * block);
  last  = graph.begin() + ((networkHostID + 1) * block);
  if (networkHostID == (networkHostNum-1)) last = graph.end();
    
 std::cout << "host = "<< networkHostID << " f  = "<< f << " l = " << l <<"\n";

  // create the nodes
   printf ("host: %u creating nodes\n", networkHostID);
   galois::for_each(first,last,create_nodes(dgraph,lk));
   //galois::for_each(graph.begin(),graph.end(),create_nodes(dgraph,lk));
}
static void create_rest_graph(Graphp dgraph) {
   //galois::for_each(graph.begin(),graph.end(), create_nodes(dgraph,lk));
   //printf ("%lu nodes in %u host with block size %lu\n", mapping.size(), networkHostID, block);
   // create the local edges
   printf ("host: %u creating local edges\n", networkHostID);
   unsigned count = 0;
   unsigned scount = 0;
   unsigned rcount = 0;

    unsigned cc = 0;
    cout<<"Number of nodes in dgraph = "<<std::distance(dgraph->begin(),dgraph->end())<<endl;
//k=0;
auto k = 0;
for(auto ii = dgraph->begin(),ee=dgraph->end(); ii!=ee; ++ii,++k) {
   cout<<"Value of k = "<<k<<endl;
    printNode(dgraph->getData(*ii));
}
   
k=0;
   rlookup.clear();
   assert(!rlookup.size());  
/* Mapping */
    
    auto dg_it = dgraph->begin(); 
    for(auto ii = graph.begin(); ii != graph.end(); ++ii) {
	mapping[*ii] = *dg_it;
	Node& graph_node = graph.getData(*ii);
	llookup[graph_node] = *dg_it;
	++dg_it;
	//if(dg_it == dgraph->end())
	  //  cout<<"ERRROR!!!!"<<endl;
    }

for(auto ii = graph.begin(); ii != graph.end(); ++ii) {
    Graph::edge_iterator vv = graph.edge_begin(*ii);
    Graph::edge_iterator ev = graph.edge_end(*ii);
    
    scount++;
    for (Graph::edge_iterator jj = vv; jj != ev; ++jj) {
      Node& node = lookup[graph.getEdgeDst(jj)];
      unsigned int edge_data = graph.getEdgeData(jj);
      
   //   if ((f <= node.ID) && (node.ID < l)) {
	//std::cout<<"I got a movie node..\n"<<std::endl;
        dgraph->addEdge(mapping[*ii],mapping[graph.getEdgeDst(jj)], edge_data);
        count++;
     // }
/* else {
        uint32_t host = node.ID/block;
        if (host == networkHostNum) --host;
        if (host > networkHostNum) {
          printf("ERROR Wrong host ID: %u\n", host);
          abort();
        }
        SendBuffer b;
        gSerialize(b, networkHostID, node);
        getSystemNetworkInterface().send(host,getRemoteNode_landing_pad,b);
        getSystemNetworkInterface().handleReceives();
        requested.insert(node);
        ++rcount;
      }
*/

    }
  }
  /* For checking if graph is correct  */
for(auto ii = dgraph->begin(); k != (num_movie_nodes); ++k, ++ii) {
      DGraph::edge_iterator edge_begin = dgraph->edge_begin(*ii);
      DGraph::edge_iterator edge_end = dgraph->edge_end(*ii);
        
	cout<<"Distance: "<<std::distance(edge_end,edge_begin)<<endl;
	if(edge_begin == edge_end)
	    std::cout << "????????????????kuch problem hai\n";
  /*    Node& movie_data = g->getData(movie);
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
}
/*for(auto ii = dgraph->begin(), ee = dgraph->end(); ii != ee; ++ii) {
    DGraph::edge_iterator ie = dgraph->edge_begin(*ii);
    DGraph::edge_iterator ei = dgraph->edge_end(*ii);
    cout<<"Distance: "<<std::distance(ie,ei)<<endl;
}*/
    

  //std::cout <<"host="<<networkHostID<<"cc = " <<cc<<"\n";
    
  printf ("host: %u nodes %u and edges %u remote edges %u\n", networkHostID, scount, count, rcount);
  printf ("host: %u done creating local edges\n", networkHostID);
 /* uint64_t recvsize=0, reqsize;
  reqsize = requested.size();
  do {
    getSystemNetworkInterface().handleReceives();
    slock.lock();
    recvsize = rlookup.size();
    slock.unlock();
    if (recvsize > reqsize) {
    printf("Aborting..\n");
      abort();
    }
  } while(recvsize != reqsize);
    printf("Host:%u reached here...\n",networkHostID);
  //program_barrier();

  printf ("host: %u creating remote edges\n", networkHostID);
  for(auto ii = first; ii != last; ++ii) {
    Graph::edge_iterator vv = graph.edge_begin(*ii, galois::MethodFlag::NONE);
    Graph::edge_iterator ev = graph.edge_end(*ii, galois::MethodFlag::NONE);
    for (Graph::edge_iterator jj = vv; jj != ev; ++jj) {
      Node& node = lookup[graph.getEdgeDst(jj)];
      unsigned int edge_data = graph.getEdgeData(jj);
      if (!((f <= node.ID) && (node.ID < l))) {
        dgraph->addEdge(mapping[*ii],rlookup[node], edge_data);
      }
    }
  }
  printf ("host: %u done creating remote edges\n", networkHostID);
*/
 // program_barrier();
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
    printf("0..Done creating dist graph..\n");
	
}



void readGraph(Graphp dgraph, std::string inputFile) {
   
    readInputGraph(dgraph, inputFile); 

}





int main(int argc, char** argv)
{	
	
	galois::StatManager statManager;
	galois::runtime::networkStart();
	if(argc < 3)
	{
		std::cout << "Usage: <input binary gr file> <thread count>" << std::endl;
		return -1;
	}
	
	std::cout<< "start reading and building Graph on Host " <<networkHostID<<"\n";
	//read command line parameters
	//const char* inputFile = argv[1];
	std::string inputFile = argv[1];
	unsigned int threadCount = atoi(argv[2]);

	//how many threads Galois should use
	galois::setActiveThreads(threadCount);

	//prints out the number of conflicts at the end of the program

	Graphp dgraph = DGraph::allocate();

	galois::StatTimer Tinitial("Initialization Time");
	Tinitial.start();
	readGraph(dgraph, inputFile);    
	Tinitial.stop();
	create_rest_graph(dgraph);	
	std::cout<< "Done reading and building Graph\n";
	
	std::cout<< "Running Verification\n";
	//verify(dgraph);

    
	std::cout << "num_movie_nodes = " << num_movie_nodes << "\n";
	//program_barrier();
	galois::StatTimer T("Sgd Time");
	T.start();
	//sgd_algo()(dgraph);
	 T.stop();   
	std::cout << "===========calling shared sgd num_movie_nodes = " << num_movie_nodes << "\n";
	//program_barrier();
	galois::StatTimer T2("shared Sgd Time");
	T2.start();
	//shared_sgd_algo()();
	 T2.stop();   



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
                         galois::wl<galois::worklists::OrderedByIntegerMetric
                         <projCount, galois::worklists::dChunkedLIFO<32>>>());

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
