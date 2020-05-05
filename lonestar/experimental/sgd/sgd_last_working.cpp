/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
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

#include "galois/Galois.h"
#include "galois/Graph/Graph.h"
#include "galois/Graph/LCGraph.h"
#include "galois/ParallelSTL/ParallelSTL.h"
#include "llvm//Support/CommandLine.h"
#include "Lonestar/BoilerPlate.h"
#include "galois/runtime/Network.h"
#include "galois/Timer.h"
#include "galois/Timer.h"

// Distributed Galois
#include "galois/graphs/Graph3.h"
#include "galois/runtime/DistSupport.h"

#include <boost/iterator/transform_iterator.hpp>

typedef struct Node {
  double* latent_vector; // latent vector to be learned
  unsigned int
      updates; // number of updates made to this node (only used by movie nodes)
  unsigned int edge_offset; // if a movie's update is interrupted, where to
                            // start when resuming
  unsigned int ID;
  bool operator==(const Node& other) const { return (ID == other.ID); }
  bool operator<(const Node& other) const { return (ID < other.ID); }
} Node;

namespace std {

template <>
struct hash<Node> {
  std::size_t operator()(const Node& other) const {
    using std::hash;
    using std::size_t;
    return (hash<unsigned int>()(other.ID));
  }
};
} // namespace std

// local computation graph (can't add nodes/edges at runtime)
// node data is Node, edge data is unsigned int... [movie--->user]

typedef galois::graphs::LC_Numa_Graph<Node, unsigned int> Graph;
typedef Graph::GraphNode GNode;
Graph graph;

// Distributed Graph Nodes.
typedef galois::graphs::ThirdGraph<Node, unsigned int,
                                   galois::graphs::EdgeDirection::Out>
    DGraph;
typedef DGraph::NodeHandle DGNode;
typedef typename DGraph::pointer Graphp;

// TODO : replace maps with unordered_map
std::unordered_map<GNode, Node> lookup;
std::unordered_map<GNode, DGNode> mapping;
std::unordered_map<Node, DGNode> llookup;
std::unordered_map<Node, DGNode> rlookup;
std::set<Node> requested;

volatile unsigned prog_barrier = 0;
// std::atomic<unsigned> prog_barrier;
unsigned int num_movie_nodes = 0;

using namespace galois::runtime;
typedef galois::runtime::LL::SimpleLock<true> SLock;
SLock slock;
SLock pblock;

unsigned int LATENT_VECTOR_SIZE = 20;
double LEARNING_RATE            = 0.001;
double DECAY_RATE               = 0.9;
double LAMBDA                   = 0.001;
unsigned int MAX_MOVIE_UPDATES  = 5;
unsigned int NUM_RATINGS        = 0;
static const double MINVAL      = -1e+100;
static const double MAXVAL      = 1e+100;

double vector_dot(const Node& movie_data, const Node& user_data) {
  const double* __restrict__ movie_latent = movie_data.latent_vector;
  const double* __restrict__ user_latent  = user_data.latent_vector;

  double dp = 0.0;
  for (int i = 0; i < LATENT_VECTOR_SIZE; ++i)
    dp += user_latent[i] * movie_latent[i];
  assert(std::isnormal(dp));
  return dp;
}

double calcPrediction(const Node& movie_data, const Node& user_data) {
  double pred = vector_dot(movie_data, user_data);
  pred        = std::min(MAXVAL, pred);
  pred        = std::max(MINVAL, pred);
  return pred;
}

inline void doGradientUpdate(Node& movie_data, Node& user_data,
                             unsigned int edge_rating) {
  double* __restrict__ movie_latent = movie_data.latent_vector;
  double step_size                  = LEARNING_RATE * 1.5 /
                     (1.0 + DECAY_RATE * pow(movie_data.updates + 1, 1.5));
  double* __restrict__ user_latent = user_data.latent_vector;

  double cur_error = edge_rating - vector_dot(movie_data, user_data);

  for (unsigned int i = 0; i < LATENT_VECTOR_SIZE; i++) {
    double prev_movie_val = movie_latent[i];
    double prev_user_val  = user_latent[i];
    movie_latent[i] +=
        step_size * (cur_error * prev_user_val - LAMBDA * prev_movie_val);
    user_latent[i] +=
        step_size * (cur_error * prev_movie_val - LAMBDA * prev_user_val);
  }
}

/*
inline void doGradientUpdate(Node& movie_data, Node& user_data, unsigned int
edge_rating)
{
        double step_size = LEARNING_RATE * 1.5 / (1.0 + DECAY_RATE *
pow(movie_data.updates + 1, 1.5)); double* __restrict__ movie_latent =
movie_data.latent_vector; double* __restrict__ user_latent =
user_data.latent_vector;

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

                movie_latent[i] -= step_size * (cur_error * user_latent[i] +
LAMBDA * prev_movie_val); user_latent[i] -= step_size * (cur_error *
prev_movie_val + LAMBDA * user_latent[i]);
        }
}

*/

static void progBarrier_landing_pad(RecvBuffer& buf) {
  gDeserialize(buf, prog_barrier);
  ++prog_barrier;
}

static void program_barrier() {
  SendBuffer b;
  gSerialize(b, prog_barrier);
  getSystemNetworkInterface().broadcast(progBarrier_landing_pad, b);

  // unsigned old_val = prog_barrier;
  // unsigned new_val =  ++prog_barrier;
  prog_barrier++;
  // prog_barrier.compare_exchange_strong(old_val,new_val);
  printf("Entering barrier..%d\n", prog_barrier);
  do {
    // std::cout << "inside do loop\n";
    getSystemLocalDirectory().makeProgress();
    getSystemRemoteDirectory().makeProgress();
    getSystemNetworkInterface().handleReceives();
  } while (prog_barrier != networkHostNum);

  prog_barrier = 0;
  printf("Left barrier..\n");
}

void verify(Graphp g) {
  // Checking RMSE for verification
  typedef galois::GAccumulator<double> AccumDouble;
  //  unsigned int num_movie_nodes;
  AccumDouble rms;
  // DGraph::iterator ii = g->begin() + num_movie_nodes;
  galois::do_all(g, [&g, &rms](DGNode n) {
    for (auto ii = g->edge_begin(n); ii != g->edge_end(n); ++ii) {

      DGNode m      = g->getEdgeDst(ii);
      double pred   = calcPrediction(g->getData(n), g->getData(m));
      double rating = ii->getValue();

      if (!std::isnormal(pred))
        std::cout << "Denormal Warning\n";
      rms += ((pred - rating) * (pred - rating));
    }
  });

  double total_rms      = rms.reduce();
  double normalized_rms = sqrt(total_rms / NUM_RATINGS);
  std::cout << "RMSE Total: " << total_rms << " Normalized: " << normalized_rms
            << std::endl;
}

/* Operator */

struct sgd_algo {
  galois::GAccumulator<size_t> numNodes;
  // unsigned int num_movie_nodes;

  struct Process : public galois::runtime::Lockable {
    Graphp g;
    sgd_algo* self;
    Process() {}
    // sgd(Graphp _g) : g(_g) {}
    Process(sgd_algo* s, Graphp _g) : g(_g), self(s) {}
    // void operator()(const DGNode& n, galois::UserContext<DGNode>&)
    // {(*this)(n);}
    void operator()(const DGNode& movie, galois::UserContext<DGNode>& ctx) {

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

      Node& movie_data               = g->getData(movie);
      DGraph::edge_iterator edge_it  = g->edge_begin(movie);
      DGraph::edge_iterator edge_end = g->edge_end(movie);

      std::advance(edge_it, movie_data.edge_offset);
      DGNode user     = g->getEdgeDst(edge_it);
      Node& user_data = g->getData(user);

      unsigned int edge_rating = edge_it->getValue();

      // Call the gradient routine
      doGradientUpdate(movie_data, user_data, edge_rating);

      ++edge_it;
      ++movie_data.edge_offset;

      // This is the last user
      if (edge_it == g->edge_end(movie)) // galois::MethodFlag::NONE))
      {
        // start back at the first edge again

        movie_data.edge_offset = 0;

        movie_data.updates++;
        if (movie_data.updates < MAX_MOVIE_UPDATES)
          ctx.push(movie);
      } else {
        ctx.push(movie);
      }
    }

    typedef int tt_has_serialize;
    void serialize(galois::runtime::SerializeBuffer& s) const {
      gSerialize(s, g);
    }
    void deserialize(galois::runtime::DeSerializeBuffer& s) {
      gDeserialize(s, g);
    }
  };
  void operator()(Graphp g) {
    DGraph::iterator ii = g->begin();
    // std::advance(ii,num_movie_nodes);

    Graph::iterator jj = graph.begin();
    // std::advance(jj, num_movie_nodes+1);

    Node& dg_movie = g->getData(*ii);
    Node& g_movie  = graph.getData(*jj);

    std::cout << "dg_movie = " << dg_movie.ID << "\n";
    std::cout << "g_movie = " << g_movie.ID << "\n";
    std::cout << "num movie nodes  = " << num_movie_nodes << "\n";

    // galois::for_each(g, Process(this,g), "Process");
    //	galois::for_each(g->begin(), ii, Process(this,g), "SGD Process");

    // Verification routine
    std::cout << "Running Verification after completion\n";
    verify(g);

    // program_barrier();
    // std::cout << "number of nodes = "<<numNodes.reduce() << "\n";
  }
};

struct create_nodes {
  Graphp g;
  SLock& l;
  create_nodes(Graphp _g, SLock& _l) : g(_g), l(_l) {}

  void operator()(GNode& item, galois::UserContext<GNode>& ctx) {
    Node& node = graph.getData(item, galois::MethodFlag::NONE);
    DGNode n   = g->createNode(node);
    g->addNode(n);
    l.lock();
    mapping[item]         = n;
    llookup[lookup[item]] = n;
    l.unlock();
  }
};

unsigned int initializeGraphData(Graph& g) {
  unsigned int seed = 42;
  std::default_random_engine eng(seed);
  std::uniform_real_distribution<double> random_lv_value(0, 0.1);

  unsigned int num_movie_nodes = 0;
  unsigned int num_user_nodes  = 0;
  unsigned index               = 0;
  unsigned int numRatings      = 0;
  //  for all movie and user nodes in the graph
  for (Graph::iterator i = g.begin(), end = g.end(); i != end; ++i) {
    Graph::GraphNode gnode = *i;
    Node& data             = g.getData(gnode);

    data.ID = index;
    ++index;
    data.updates = 0;

    // fill latent vectors with random values
    double* lv = new double[LATENT_VECTOR_SIZE];
    for (int i = 0; i < LATENT_VECTOR_SIZE; i++) {
      lv[i] = random_lv_value(eng);
    }
    data.latent_vector     = lv;
    unsigned int num_edges = g.edge_end(gnode, galois::MethodFlag::NONE) -
                             g.edge_begin(gnode, galois::MethodFlag::NONE);
    // std::cout << "num edges = " << num_edges << "\n";
    numRatings += num_edges;
    if (num_edges > 0)
      ++num_movie_nodes;
    else
      ++num_user_nodes;

    data.edge_offset = 0;
  }

  NUM_RATINGS = numRatings;
  return num_movie_nodes;
}

static void recvRemoteNode_landing_pad(RecvBuffer& buf) {
  DGNode n;
  // unsigned num;
  Node node;
  uint32_t host;
  gDeserialize(buf, host, node, n);
  slock.lock();
  rlookup[node] = n;
  slock.unlock();
}

static void getRemoteNode_landing_pad(RecvBuffer& buf) {
  // unsigned num;
  Node node;
  uint32_t host;
  gDeserialize(buf, host, node);
  SendBuffer b;
  slock.lock();
  gSerialize(b, networkHostID, node, llookup[node]);
  slock.unlock();
  getSystemNetworkInterface().send(host, recvRemoteNode_landing_pad, b);
}

static void create_dist_graph(Graphp dgraph, std::string inputFile) {
  SLock lk;
  prog_barrier = 0;
  uint64_t block, f, l;
  Graph::iterator first, last;
  graph.structureFromFile(inputFile);
  num_movie_nodes = initializeGraphData(graph);
  std::cout << "Number of movie nodes=" << num_movie_nodes << std::endl;
  unsigned size = 0;
  for (auto ii = graph.begin(); ii != graph.end(); ++ii) {
    lookup[*ii] = graph.getData(*ii);
    ++size;
  }
  std::cout << "Size=" << size << std::endl;

  block = size / networkHostNum;
  f     = networkHostID * block;
  l     = (networkHostID + 1) * block;
  first = graph.begin() + (networkHostID * block);
  last  = graph.begin() + ((networkHostID + 1) * block);
  if (networkHostID == (networkHostNum - 1))
    last = graph.end();

  std::cout << "host = " << networkHostID << " f  = " << f << " l = " << l
            << "\n";

  // create the nodes
  printf("host: %u creating nodes\n", networkHostID);
  galois::for_each(first, last, create_nodes(dgraph, lk));
  printf("%lu nodes in %u host with block size %lu\n", mapping.size(),
         networkHostID, block);
  // create the local edges
  printf("host: %u creating local edges\n", networkHostID);
  unsigned count  = 0;
  unsigned scount = 0;
  unsigned rcount = 0;

  unsigned cc = 0;

  rlookup.clear();
  assert(!rlookup.size());
  for (auto ii = first; ii != last; ++ii) {
    Graph::edge_iterator vv = graph.edge_begin(*ii, galois::MethodFlag::NONE);
    Graph::edge_iterator ev = graph.edge_end(*ii, galois::MethodFlag::NONE);

    // if(vv != ev)
    //{
    //	++cc;
    //  }
    scount++;
    for (Graph::edge_iterator jj = vv; jj != ev; ++jj) {
      Node& node             = lookup[graph.getEdgeDst(jj)];
      unsigned int edge_data = graph.getEdgeData(jj);

      if ((f <= node.ID) && (node.ID < l)) {
        // std::cout<<"I got a movie node..\n"<<std::endl;
        dgraph->addEdge(mapping[*ii], mapping[graph.getEdgeDst(jj)], edge_data);
        count++;
      } else {
        uint32_t host = node.ID / block;
        if (host == networkHostNum)
          --host;
        if (host > networkHostNum) {
          printf("ERROR Wrong host ID: %u\n", host);
          abort();
        }
        SendBuffer b;
        gSerialize(b, networkHostID, node);
        getSystemNetworkInterface().send(host, getRemoteNode_landing_pad, b);
        getSystemNetworkInterface().handleReceives();
        requested.insert(node);
        ++rcount;
      }
    }
  }
  // std::cout <<"host="<<networkHostID<<"cc = " <<cc<<"\n";

  printf("host: %u nodes %u and edges %u remote edges %u\n", networkHostID,
         scount, count, rcount);
  printf("host: %u done creating local edges\n", networkHostID);
  uint64_t recvsize = 0, reqsize;
  reqsize           = requested.size();
  do {
    getSystemNetworkInterface().handleReceives();
    slock.lock();
    recvsize = rlookup.size();
    slock.unlock();
    if (recvsize > reqsize) {
      printf("Aborting..\n");
      abort();
    }
  } while (recvsize != reqsize);
  printf("Host:%u reached here...\n", networkHostID);
  // program_barrier();

  printf("host: %u creating remote edges\n", networkHostID);
  for (auto ii = first; ii != last; ++ii) {
    Graph::edge_iterator vv = graph.edge_begin(*ii, galois::MethodFlag::NONE);
    Graph::edge_iterator ev = graph.edge_end(*ii, galois::MethodFlag::NONE);
    for (Graph::edge_iterator jj = vv; jj != ev; ++jj) {
      Node& node             = lookup[graph.getEdgeDst(jj)];
      unsigned int edge_data = graph.getEdgeData(jj);
      if (!((f <= node.ID) && (node.ID < l))) {
        dgraph->addEdge(mapping[*ii], rlookup[node], edge_data);
      }
    }
  }
  printf("host: %u done creating remote edges\n", networkHostID);

  // program_barrier();
}

static void readInputGraph_landing_pad(RecvBuffer& buf) {
  Graphp dgraph;
  std::string inputFile;
  gDeserialize(buf, inputFile, dgraph);
  create_dist_graph(dgraph, inputFile);
  printf("1..Done creating dist graph..\n");
}

void readInputGraph(Graphp dgraph, std::string inputFile) {
  std::cout << "NetworkHostNum=" << networkHostNum << std::endl;
  if (networkHostNum > 1) {
    SendBuffer b;
    gSerialize(b, inputFile, dgraph);
    getSystemNetworkInterface().broadcast(readInputGraph_landing_pad, b);
    printf("Handling receives...\n");
    getSystemNetworkInterface().handleReceives();
    printf("Done Handling receives...\n");
  }

  create_dist_graph(dgraph, inputFile);
  printf("0..Done creating dist graph..\n");
}

void readGraph(Graphp dgraph, std::string inputFile) {

  readInputGraph(dgraph, inputFile);
}

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cout << "Usage: <input binary gr file> <thread count>" << std::endl;
    return -1;
  }

  std::cout << "start reading and building Graph\n";
  // read command line parameters
  // const char* inputFile = argv[1];
  std::string inputFile    = argv[1];
  unsigned int threadCount = atoi(argv[2]);

  // how many threads Galois should use
  galois::setActiveThreads(threadCount);

  // prints out the number of conflicts at the end of the program
  galois::StatManager statManager;
  galois::runtime::networkStart();

  Graphp dgraph = DGraph::allocate();

  galois::StatTimer Tinitial("Initialization Time");
  Tinitial.start();
  readGraph(dgraph, inputFile);
  Tinitial.stop();

  std::cout << "Done reading and building Graph\n";

  std::cout << "Running Verification\n";
  verify(dgraph);

  std::cout << "num_movie_nodes = " << num_movie_nodes << "\n";
  // program_barrier();
  galois::StatTimer T("Sgd Time");
  T.start();
  sgd_algo()(dgraph);
  T.stop();

  // allocate local computation graph
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

      //galois::for_each<>(boost::counting_iterator<int>(0),
     boost::counting_iterator<int>(100), op(g));

      //galois::for_each<>(f.begin(), f.end(), AddNodes(g,f));

      std::cout << "Done making graph\n";


      galois::StatTimer timer;
      timer.start();

      //do the SGD computation in parallel
      //the initial worklist contains all the movie nodes
      //the movie nodes are located at the top num_movie_nodes nodes in the
     graph
      //the worklist is a priority queue ordered by the number of updates done
     to a movie
      //the projCount functor provides the priority function on each node
      //Graphp::iterator ii = g.begin();
      //std::advance(ii,num_movie_nodes); //advance moves passed in iterator
      galois::for_each(g.begin(), ii, sgd(g),
                           galois::wl<galois::worklists::OrderedByIntegerMetric
                           <projCount,
     galois::worklists::PerSocketChunkLIFO<32>>>());

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
