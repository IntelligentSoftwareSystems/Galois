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
 * Stochastic gradient descent for matrix factorization, implemented with Galois.
 *
 * Author: Prad Nelluru <pradn@cs.utexas.edu>
 */

#include <iostream>
#include <random>
#include <algorithm>
#include <cmath>
#include <cstdlib>

#include "Galois/Accumulator.h"
#include "Galois/Galois.h"
#include "Galois/Timer.h"
#include "Galois/Statistic.h"

#include "Galois/Graphs/Graph.h"
#include "Galois/Graphs/LCGraph.h"

#include "Galois/Substrate/PaddedLock.h"

#include "Lonestar/BoilerPlate.h"

static const char* const name = "Matrix Completion";
static const char* const desc = "Computes Matrix Decomposition using Stochastic Gradient Descent";
static const char* const url = 0;

static const int LATENT_VECTOR_SIZE = 20; //Prad's default: 100, Intel: 20
static const unsigned int MAX_MOVIE_UPDATES = 1; //Prad's default: 5
static const double MINVAL = -1e+100;
static const double MAXVAL = 1e+100;

static const double LEARNING_RATE = 0.001; // GAMMA, Purdue: 0.01 Intel: 0.001
static const double DECAY_RATE = 0.9; // STEP_DEC, Purdue: 0.1 Intel: 0.9
static const double LAMBDA = 0.001; // Purdue: 1.0 Intel: 0.001
static const double BottouInit = 0.1;

enum Algo {
  nodeMovie,
  nodeMoviePri,
  edgeMovie,
  block,
  blockAndSliceUsers,
  blockAndSliceBoth,
  sliceMarch,
  sliceJump
};

enum Learn {
  Intel,
  Purdue,
  Bottou,
  Inv
};

namespace cll = llvm::cl;
static cll::opt<std::string> inputFile (cll::Positional, cll::desc("<input file>"), cll::Required);
static cll::opt<unsigned> usersPerBlockSlice ("usersPerBlk", cll::desc ("[users per block slice]"), cll::init (2048));
static cll::opt<unsigned> moviesPerBlockSlice ("moviesPerBlk", cll::desc ("[movies per block slice]"), cll::init (350));
static cll::opt<bool> verifyPerIter ("verifyPerIter", cll::desc ("compute RMS every iter"), cll::init (false));

static cll::opt<int> shiftFactor("shiftRating", cll::desc("Shift ratings down by"), cll::init(0));

static cll::opt<Algo> algo(cll::desc("Choose an algorithm:"),
                           cll::values(
                                       clEnumVal(nodeMovie, "Node by Movies"),
                                       clEnumVal(edgeMovie, "Edge by Movies"),
                                       clEnumVal(nodeMoviePri, "Node delta error"),
                                       clEnumVal(block, "Block by Users and Movies"),
                                       clEnumVal(blockAndSliceUsers, "Block by Users and Movies, Slice by Users"),
                                       clEnumVal(blockAndSliceBoth, "Block by Users and Movies, Slice by Users and Movies (default)"),
                                       clEnumVal(sliceMarch, "Marching Slices version"),
                    clEnumVal(sliceJump, "Jumping Slices version"),
                                       clEnumValEnd), 
                           cll::init(blockAndSliceBoth));

static cll::opt<Learn> learn(cll::desc("Choose a learning function:"),
                             cll::values(
                                         clEnumVal(Intel, "Intel"),
                                         clEnumVal(Purdue, "Perdue"),
                                         clEnumVal(Bottou, "Bottou"),
                                         clEnumVal(Inv, "Simple Inverse"),
                                         clEnumValEnd), 
                             cll::init(Intel));



struct Node {
  double latent_vector[LATENT_VECTOR_SIZE]; //latent vector to be learned
  unsigned int updates; //number of updates made to this node (only used by movie nodes)
  unsigned int edge_offset; //if a movie's update is interrupted, where to start when resuming

  void dump(std::ostream& os) {
    os << "{" << latent_vector[0];
    for (int i = 1; i < LATENT_VECTOR_SIZE; ++i)
      os << ", " << latent_vector[i];
    os << "}";
  }
};

//local computation graph (can't add nodes/edges at runtime)
//node data is Node, edge data is unsigned int
// typedef Galois::Graph::LC_CSR_Graph<Node, unsigned int> Graph;

typedef typename Galois::Graph::LC_CSR_Graph<Node, int>
::with_numa_alloc<true>::type
::with_no_lockable<false>::type Graph;
typedef Graph::GraphNode GNode;

struct ThreadWorkItem {
  unsigned int movieRangeStart;
  unsigned int movieRangeEnd;
  unsigned int userRangeStart;
  unsigned int userRangeEnd;
  unsigned int usersPerBlockSlice;
  unsigned int moviesPerBlockSlice;

  unsigned int sliceStart; //only used in march variation
  unsigned int numSlices; //only used in march variation

  //debug
  unsigned int id;
  unsigned int updates;
  double timeTaken;
};

unsigned int NUM_MOVIE_NODES = 0;
unsigned int NUM_USER_NODES = 0;
unsigned int NUM_RATINGS = 0;

//possibly over-typed
double vector_dot(const Node& movie_data, const Node& user_data) {
  //Could just specify restrict on parameters since vector is built in
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
  double p = pred;
  pred = std::min (MAXVAL, pred);
  pred = std::max (MINVAL, pred);
  if (p != pred)
    std::cerr << "clamped " << p << " to " << pred << "\n";
  return pred;
}

struct LearnFN {
  virtual double step_size(unsigned int round) const = 0;
};

struct PurdueLearnFN : public LearnFN {
  virtual double step_size(unsigned int round) const {
    return LEARNING_RATE * 1.5 / (1.0 + DECAY_RATE * pow(round + 1, 1.5));
  }
};

struct IntelLearnFN : public LearnFN {
  virtual double step_size(unsigned int round) const {
    return LEARNING_RATE * pow (DECAY_RATE, round);
  }
};

struct BottouLearnFN : public LearnFN {
  virtual double step_size(unsigned int round) const {
    return BottouInit / (1 + BottouInit*LAMBDA*round);
  }
};

struct InvLearnFN : public LearnFN {
  virtual double step_size(unsigned int round) const {
    return (double)1 / (double)(round + 1);
  }
};

double doGradientUpdate(Node& movie_data, Node& user_data, int edge_rating, double step_size)
{
  double* __restrict__ movie_latent = movie_data.latent_vector;
  double* __restrict__ user_latent = user_data.latent_vector;
  
  //calculate error
  double old_dp = vector_dot(movie_data, user_data);
  double cur_error = edge_rating - old_dp;
  assert(cur_error < 1000 && cur_error > -1000);
  
  //take gradient step
  for(unsigned int i = 0; i < LATENT_VECTOR_SIZE; i++)
    {
      double prev_movie_val = movie_latent[i];
      double prev_user_val = user_latent[i];
      movie_latent[i] += step_size * (cur_error * prev_user_val  - LAMBDA * prev_movie_val);
      assert(std::isnormal(movie_latent[i]));
      user_latent[i]  += step_size * (cur_error * prev_movie_val - LAMBDA * prev_user_val);
      assert(std::isnormal(user_latent[i]));
    }
  return cur_error;
}

void verify (Graph& g) {
  // computing Root Mean Square Error
  // Assuming only movie nodes have edges

  typedef Galois::GAccumulator<double> AccumDouble;

  AccumDouble rms;

  Galois::do_all_local (g, 
      [&g, &rms] (GNode n) {
      for (auto e = g.edge_begin (n, Galois::MethodFlag::UNPROTECTED)
        , e_end = g.edge_end (n, Galois::MethodFlag::UNPROTECTED); e != e_end; ++e) {

      GNode m = g.getEdgeDst (e);
      double pred = calcPrediction (g.getData (n, Galois::MethodFlag::UNPROTECTED), g.getData (m, Galois::MethodFlag::UNPROTECTED));
      double rating = g.getEdgeData (e, Galois::MethodFlag::UNPROTECTED);
      
                        if (!std::isnormal(pred))
                          std::cout << "denormal warning\n";
                        if (pred > 100.0 || pred < -100.0)
                          std::cout << "Big difference " << pred << " should be " << rating << "\n";

      rms += ((pred - rating) * (pred - rating));
      }
      });

  double total_rms = rms.reduce ();
  double normalized_rms = sqrt(total_rms/NUM_RATINGS);
  
  std::cout << "RMSE Total: " << total_rms << " Normalized: " << normalized_rms << std::endl;
}

//Simple by-movie node-based
struct sgd_node_movie {
  Graph& g;
  double step_size;
  sgd_node_movie(Graph& g, double ss) :g(g), step_size(ss) {}

  template<typename Context>
  void operator()(GNode node, Context& cnx) {
    for (auto ii = g.edge_begin (node), ee = g.edge_end (node);
         ii != ee; ++ii)
      doGradientUpdate(g.getData(node), g.getData(g.getEdgeDst(ii)), g.getEdgeData(ii), step_size);
  }

  static void go(Graph& g, unsigned int numMovieNodes, unsigned int numUserNodes, const LearnFN* lf) {
    std::deque<GNode> Movies;
    for (auto ii = g.begin(), ee = g.end(); ii != ee; ++ii)
      if (g.edge_begin(*ii) != g.edge_end(*ii))
        Movies.push_back(*ii);
    for (int i = 0; i < 10; ++i) {
      if (verifyPerIter)
        verify(g);
      double step_size = lf->step_size(i);
      std::cout << "Step Size: " << step_size << "\n";
      // if (i != 0)
      //   std::random_shuffle(Movies.begin(), Movies.end());
      Galois::for_each(Movies.begin(), Movies.end(), sgd_node_movie(g, step_size));
    }
  }
};

//priority by-movie node-based
struct sgd_node_movie_pri {
  Graph& g;
  double step_size;
  sgd_node_movie_pri(Graph& g, double ss) :g(g), step_size(ss) {}

  template<typename Context>
  void operator()(GNode node, Context& cnx) {
    for (auto ii = g.edge_begin (node), ee = g.edge_end (node);
         ii != ee; ++ii) {
      double e1 = doGradientUpdate(g.getData(node), g.getData(g.getEdgeDst(ii)), g.getEdgeData(ii), step_size);
      double e2 = g.getEdgeData(ii) - calcPrediction(g.getData(node), g.getData(g.getEdgeDst(ii)));
      if (std::abs(e1 - e2) > 20) {
        std::cerr << "A" << std::abs(e1 - e2);
        cnx.push(g.getEdgeDst(ii));
      }
    }
  }

  static void go(Graph& g, unsigned int numMovieNodes, unsigned int numUserNodes, const LearnFN* lf) {
    std::deque<GNode> Movies;
    for (auto ii = g.begin(), ee = g.end(); ii != ee; ++ii)
      if (std::distance(g.edge_begin(*ii), g.edge_end(*ii)) > 100)
        Movies.push_back(*ii);
    for (int i = 0; i < 10; ++i) {
      if (verifyPerIter)
        verify(g);
      double step_size = lf->step_size(i);
      std::cout << "Step Size: " << step_size << "\n";
      // if (i != 0)
      //   std::random_shuffle(Movies.begin(), Movies.end());
      Galois::for_each(Movies.begin(), Movies.end(), sgd_node_movie_pri(g, step_size), 
                       Galois::wl<Galois::WorkList::dChunkedFIFO<>>());
    }
  }
};

//Simple by-edge grouped by movie (only one edge per movie on the WL at any time)
struct sgd_edge_movie {
  Graph& g;
  double step_size;
  sgd_edge_movie(Graph& g, double ss) :g(g), step_size(ss) {}

  template<typename Context>
  void operator()(GNode node, Context& cnx) {
    auto ii = g.edge_begin (node, Galois::MethodFlag::UNPROTECTED), ee = g.edge_end (node, Galois::MethodFlag::UNPROTECTED);
    if (ii == ee) return;
    auto& nd = g.getData(node);
    std::advance(ii, nd.edge_offset);
    auto& no = g.getData(g.getEdgeDst(ii));
    doGradientUpdate(nd, no, g.getEdgeData(ii), step_size);
    ++nd.edge_offset;
    ++ii;
    if (ii == ee) { nd.edge_offset = 0; return; }
    else { cnx.push(node); }
  }

  static void go(Graph& g, unsigned int numMovieNodes, unsigned int numUserNodes, const LearnFN* lf) {
    std::deque<GNode> Movies;
    for (auto ii = g.begin(), ee = g.end(); ii != ee; ++ii)
      if (g.edge_begin(*ii) != g.edge_end(*ii))
        Movies.push_back(*ii);
    for (int i = 0; i < 20; ++i) {
      if (verifyPerIter)
        verify(g);
      double step_size = lf->step_size(i);
      std::cout << "Step Size: " << step_size << "\n";
      if (i != 0)
        std::random_shuffle(Movies.begin(), Movies.end());
      Galois::for_each(Movies.begin(), Movies.end(), sgd_edge_movie(g, step_size), Galois::wl<Galois::WorkList::dChunkedLIFO<8>>());
    }
  }
};

struct sgd_block
{
  Graph& g;
  double step_size;
  sgd_block(Graph& g, double ss) : g(g), step_size(ss) {}

  void operator()(ThreadWorkItem& workItem) const
  {
    Galois::Timer timer;
    timer.start();
    int updates = 0;

    unsigned int userRangeEnd = workItem.userRangeEnd;

    //set up movie iterators
    Graph::iterator movie_it = g.begin();
    std::advance(movie_it, workItem.movieRangeStart);
    Graph::iterator end_movie_it = g.begin();
    std::advance(end_movie_it, workItem.movieRangeEnd);

    unsigned prev_updates = updates;
    unsigned last_edge_reached = 0;

    //for each movie in the range
    for(; movie_it != end_movie_it; ++movie_it)
    {
      //get movie data
      GNode movie = *movie_it;
      Node& movie_data = g.getData(movie);

      //for each edge in the range
      Graph::edge_iterator edge_it = g.edge_begin(movie, Galois::MethodFlag::UNPROTECTED) + movie_data.edge_offset;
      Graph::edge_iterator edge_end = g.edge_end(movie, Galois::MethodFlag::UNPROTECTED);
      if(edge_it == edge_end) last_edge_reached++;
      for(;edge_it != edge_end; ++edge_it, ++movie_data.edge_offset)
      {
        GNode user = g.getEdgeDst(edge_it);

        //stop when you're outside the current block's user range
        if(user > userRangeEnd)
          break;

        Node& user_data = g.getData(user, Galois::MethodFlag::UNPROTECTED);
        int edge_rating = g.getEdgeData(edge_it, Galois::MethodFlag::UNPROTECTED);  

        //do gradient step
        doGradientUpdate(movie_data, user_data, edge_rating, step_size);
                                ++movie_data.updates;

        updates++;
      }

      //we just looked at the last user
      if(userRangeEnd == NUM_USER_NODES)
      {
        //start back at the first edge
        movie_data.edge_offset = 0;
      }
    }

    timer.stop();
    workItem.timeTaken = timer.get_usec();
    workItem.updates = updates;
  }

};

struct sgd_block_users
{
  Graph& g;
  double step_size;
  sgd_block_users(Graph& g, double ss) : g(g), step_size(ss) {}

  void operator()(ThreadWorkItem& workItem) const
  {
    Galois::Timer timer;
    timer.start();
    int updates = 0;

    unsigned int usersPerBlockSlice = workItem.usersPerBlockSlice;
    unsigned int currentBlockSliceEnd = workItem.userRangeStart;
    unsigned int userRangeEnd = workItem.userRangeEnd;

    while(currentBlockSliceEnd < userRangeEnd)
    {
      currentBlockSliceEnd += usersPerBlockSlice;
      if(currentBlockSliceEnd > userRangeEnd)
        currentBlockSliceEnd = userRangeEnd;

      //set up movie iterators
      Graph::iterator movie_it = g.begin();
      std::advance(movie_it, workItem.movieRangeStart);
      Graph::iterator end_movie_it = g.begin();
      std::advance(end_movie_it, workItem.movieRangeEnd);

      unsigned prev_updates = updates;
      unsigned last_edge_reached = 0;

      //for each movie in the range
      for(; movie_it != end_movie_it; ++movie_it)
      {
        //get movie data
        GNode movie = *movie_it;
        Node& movie_data = g.getData(movie);

        unsigned int currentBlockSliceEndUserId = currentBlockSliceEnd + NUM_MOVIE_NODES;// + 1;

        //for each edge in the range
        Graph::edge_iterator edge_it = g.edge_begin(movie, Galois::MethodFlag::UNPROTECTED) + movie_data.edge_offset;
        Graph::edge_iterator edge_end = g.edge_end(movie, Galois::MethodFlag::UNPROTECTED);
        if(edge_it == edge_end) last_edge_reached++;
        for(;edge_it != edge_end; ++edge_it, ++movie_data.edge_offset)
        {
          GNode user = g.getEdgeDst(edge_it);

          //stop when you're outside the current block's user range
          if(user > currentBlockSliceEndUserId)
            break;

          Node& user_data = g.getData(user, Galois::MethodFlag::UNPROTECTED);
          int edge_rating = g.getEdgeData(edge_it, Galois::MethodFlag::UNPROTECTED);  

          //do gradient step
          doGradientUpdate(movie_data, user_data, edge_rating, step_size);
                                        ++movie_data.updates;

          updates++;
        }

        //we just looked at the last user
        if(currentBlockSliceEnd == NUM_USER_NODES)
        {
          //start back at the first edge
          movie_data.edge_offset = 0;
        }
      }
    }

    timer.stop();
    workItem.timeTaken = timer.get_usec();
    workItem.updates = updates;
  }

};

//movies blocked also
struct sgd_block_users_movies
{
  Graph& g;
  double step_size;
  sgd_block_users_movies(Graph& g, double ss) : g(g), step_size(ss) {}

  void operator()(ThreadWorkItem& workItem) const
  {
    Galois::Timer timer;
    timer.start();
    int updates = 0;

    unsigned int usersPerBlockSlice = workItem.usersPerBlockSlice;
    unsigned int currentBlockSliceEnd = workItem.userRangeStart;
    unsigned int userRangeEnd = workItem.userRangeEnd;

    while(currentBlockSliceEnd < userRangeEnd)
    {
      currentBlockSliceEnd += usersPerBlockSlice;
      if(currentBlockSliceEnd > userRangeEnd)
        currentBlockSliceEnd = userRangeEnd;

      unsigned int currentMovieSliceEnd = workItem.movieRangeStart;
      unsigned int moviesPerBlockSlice = workItem.moviesPerBlockSlice;
      unsigned int movieRangeEnd = workItem.movieRangeEnd;

      while(currentMovieSliceEnd < movieRangeEnd)
      {
        Graph::iterator movie_it = g.begin();
        std::advance(movie_it, currentMovieSliceEnd);

        currentMovieSliceEnd += moviesPerBlockSlice;
        if(currentMovieSliceEnd > movieRangeEnd)
          currentMovieSliceEnd = movieRangeEnd;

        Graph::iterator end_movie_it = g.begin();
        std::advance(end_movie_it, currentMovieSliceEnd);

        //for each movie in the range
        for(; movie_it != end_movie_it; ++movie_it)
        {
          //get movie data
          GNode movie = *movie_it;
          Node& movie_data = g.getData(movie);

          unsigned int currentBlockSliceEndUserId = currentBlockSliceEnd + NUM_MOVIE_NODES;// + 1;

          //for each edge in the range
          Graph::edge_iterator edge_it = g.edge_begin(movie, Galois::MethodFlag::UNPROTECTED) + movie_data.edge_offset;
          Graph::edge_iterator edge_end = g.edge_end(movie, Galois::MethodFlag::UNPROTECTED);
          for(;edge_it != edge_end; ++edge_it, ++movie_data.edge_offset)
          {
            GNode user = g.getEdgeDst(edge_it);

            //stop when you're outside the current block's user range
            if(user > currentBlockSliceEndUserId)
              break;

            Node& user_data = g.getData(user, Galois::MethodFlag::UNPROTECTED);
            int edge_rating = g.getEdgeData(edge_it, Galois::MethodFlag::UNPROTECTED);  

            //do gradient step
            doGradientUpdate(movie_data, user_data, edge_rating, step_size);
                                                ++movie_data.updates;

            updates++;
          }

          //we just looked at the last user
          if(currentBlockSliceEnd == NUM_USER_NODES)
          {
            //start back at the first edge
            movie_data.edge_offset = 0;
          }
        }
      }
    }

    timer.stop();
    workItem.timeTaken = timer.get_usec();
    workItem.updates = updates;
  }

};

unsigned int userIdToUserNode(unsigned int userId)
{
  return userId + NUM_MOVIE_NODES + 1;
}

struct advance_edge_iterators
{
  Graph& g;
  advance_edge_iterators(Graph& g) : g(g) {}

  void operator()(ThreadWorkItem& workItem) const {
    //set up movie iterators
    Graph::iterator movie_it = g.begin();
    std::advance(movie_it, workItem.movieRangeStart);
    Graph::iterator end_movie_it = g.begin();
    std::advance(end_movie_it, workItem.movieRangeEnd);

    //for each movie in the range
    for(; movie_it != end_movie_it; ++movie_it) {
      //get movie data
      GNode movie = *movie_it;
      Node& movie_data = g.getData(movie);

      //for each edge in the range
      Graph::edge_iterator edge_it = g.edge_begin(movie, Galois::MethodFlag::UNPROTECTED) + movie_data.edge_offset;
      Graph::edge_iterator edge_end = g.edge_end(movie, Galois::MethodFlag::UNPROTECTED);
      for(;edge_it != edge_end; ++edge_it, ++movie_data.edge_offset) {
        GNode user = g.getEdgeDst(edge_it);

        //stop when you're in the acceptable user range
        if(user > userIdToUserNode(workItem.userRangeStart))
          break;
      }
    }
  }
};

//utility function to learn about a graph input
void count_ratings(Graph& g) {
  const unsigned threadCount = Galois::getActiveThreads ();

  std::vector<unsigned long> ratings_per_user(NUM_USER_NODES);
  std::vector<unsigned long> ratings_per_movie(NUM_MOVIE_NODES);

  Graph::iterator movie_it = g.begin();
  Graph::iterator end_movie_it = g.end();

  for(; movie_it != end_movie_it; ++movie_it)
  {
    //get movie data
    GNode movie = *movie_it;
    Node& movie_data = g.getData(movie);

    //for each edge in the range
    Graph::edge_iterator edge_it = g.edge_begin(movie, Galois::MethodFlag::UNPROTECTED) + movie_data.edge_offset;
    Graph::edge_iterator edge_end = g.edge_end(movie, Galois::MethodFlag::UNPROTECTED);
    for(;edge_it != edge_end; ++edge_it, ++movie_data.edge_offset)
    {
      GNode user = g.getEdgeDst(edge_it);

      ratings_per_user[user- NUM_MOVIE_NODES]++;
      ratings_per_movie[movie]++;
    }
  }

  //unsigned num_zeroes = 0;
  //for(int i = 0; i < NUM_USER_NODES; i++)
  //  {
  //  if(ratings_per_user[i] == 0) num_zeroes++;
  //std::cout << ratings_per_user[i] << std::endl;
  //}
  //std::cout << "Num zeroes " << num_zeroes << std::endl;

  unsigned int per = NUM_USER_NODES/threadCount;
  for(unsigned int i = 0; i < threadCount; i++)
  {
    unsigned start = per * i;
    unsigned end = per * (i+1);
    if(i == threadCount - 1) end = NUM_USER_NODES;

    unsigned count = 0;
    for(unsigned int j = start; j < end; j++)
      count += ratings_per_user[j];
    std::cout << i << ": " << count << std::endl;
  }

  /*unsigned num_zeroes = 0;
    for(int i = 0; i < NUM_MOVIE_NODES; i++)
    {
  //if(ratings_per_user[i] == 0) num_zeroes++;
  std::cout << ratings_per_movie[i] << std::endl;
  }*/
  //std::cout << "Num zeroes " << num_zeroes << std::endl;
}

template<typename BlockFn>
void runBlockSlices(Graph& g, const LearnFN* lf) {
  const unsigned threadCount = Galois::getActiveThreads();

  unsigned numWorkItems = threadCount;
  ThreadWorkItem workItems[numWorkItems];

  unsigned int moviesPerThread = NUM_MOVIE_NODES / numWorkItems;
  unsigned int usersPerThread = NUM_USER_NODES / numWorkItems;

  unsigned int userRangeStartPoints[numWorkItems];
  unsigned int userRangeEndPoints[numWorkItems];

  // set up initial work ranges for each thread
  for (unsigned int i = 0; i < numWorkItems; i++) {
    ThreadWorkItem wi;
    wi.movieRangeStart = moviesPerThread * i;
    wi.userRangeStart = usersPerThread * i;

    // stored for the advance_edge_iterators step
    userRangeStartPoints[i] = wi.userRangeStart;

    // last blocks take the rest
    if (i == numWorkItems - 1) {
      wi.movieRangeEnd = NUM_MOVIE_NODES;
      wi.userRangeEnd = NUM_USER_NODES;
    } else {
      wi.movieRangeEnd = wi.movieRangeStart + moviesPerThread;
      wi.userRangeEnd = (i + 1) * usersPerThread;
    }

    // stored to make it easy to move the blocks assigned to threads
    userRangeEndPoints[i] = wi.userRangeEnd;

    wi.usersPerBlockSlice = usersPerBlockSlice;
    wi.moviesPerBlockSlice = moviesPerBlockSlice;

    // debug vars
    wi.id = i;
    wi.updates = 0;

    workItems[i] = wi;
  }

  // move the edge iterators of each movie to the start of the current block
  // advances the edge iterator until it reaches the userRangeStart field of 
  // the ThreadWorkItem
  // userRangeStart isn't needed after this point
  Galois::do_all(workItems + 0, workItems + numWorkItems, advance_edge_iterators(g));

  unsigned long** updates = new unsigned long*[numWorkItems];
  for (int i = 0; i < numWorkItems; i++) {
    updates[i] = new unsigned long[numWorkItems];
  }

  // update all movies/users MAX_MOVIE_UPDATES times
  for(unsigned int update = 0; update < MAX_MOVIE_UPDATES; update++) {  
    //std::cout << "Iteration " << update << std::endl;
    if (verifyPerIter) {
      std::cout << "Step size: " << lf->step_size(update) << "\n";
      verify(g);
    }

    // work on the current blocks, move the block a thread works on to the right
    for (unsigned int j = 0; j < numWorkItems; j++) {  
      // std::cout << "Update " << update << " Block " << j << std::endl;
      // assign one ThreadWorkItem to each thread statically
      Galois::do_all(workItems + 0, workItems + numWorkItems, 
                     BlockFn(g, lf->step_size(update)));

      // move each thread's assignment of work one block to the right
      // (for the same movie nodes, look at the next range of user nodes)
      for(unsigned int k = 0; k < numWorkItems; k++) {
        ThreadWorkItem& wi = workItems[k];
        //std::cout << " (" << wi.userRangeStart << "," << wi.userRangeEnd << ") " << (long) (wi.timeTaken/1000) << "/" << (long) wi.updates;
        //std::cout << (long) wi.updates << " ";
        unsigned int column = (j + k)%numWorkItems;
        //updates[k][column] = wi.updates;
        updates[k][column] = wi.timeTaken/1000;
        //std::cout << (long) wi.updates << " ";
        //std::cout << (long) wi.timeTaken << " ";
        unsigned int nextColumn = (j+1+k)%numWorkItems;
        wi.userRangeStart = userRangeStartPoints[nextColumn];
        wi.userRangeEnd = userRangeEndPoints[nextColumn];
      }
      //std::cout << std::endl;
    }
  }

  for(int i = 0; i < numWorkItems; i++) {
    for(int j = 0; j < numWorkItems; j++)
      std::cout << updates[i][j] << " ";
    std::cout << std::endl;
  }
}

typedef Galois::Substrate::PaddedLock<true> SpinLock;

struct sgd_march
{
  Graph& g;
  SpinLock* locks; 
  double step_size;
  sgd_march(Graph& g, SpinLock* locks, double ss) : g(g), locks(locks), step_size(ss) {}

  void operator()(ThreadWorkItem& workItem) const
  {
    Galois::Timer timer;
    timer.start();
    unsigned int updates = 0;
    unsigned int conflicts = 0;

    unsigned int usersPerBlockSlice = workItem.usersPerBlockSlice;
    unsigned int currentBlockSliceEnd = workItem.userRangeStart;
    unsigned int userRangeEnd = workItem.userRangeEnd;

    unsigned int currentSliceId = workItem.sliceStart;
    unsigned int sliceUpdates = 0;

    while(sliceUpdates < MAX_MOVIE_UPDATES * workItem.numSlices)
    {
      //spinlock_lock(locks[currentSliceId]);

      if(!locks[currentSliceId].try_lock ())
      {
        conflicts++;
        locks[currentSliceId].lock ();
      }

      //if(workItem.id == 0)
      //  printf("Currslice %d Slice from %d to ", currentSliceId, currentBlockSliceEnd);

      currentBlockSliceEnd += usersPerBlockSlice;
      if(currentBlockSliceEnd > userRangeEnd)
        currentBlockSliceEnd = userRangeEnd;

      //if(workItem.id == 0)
      //  printf("%d\n", currentBlockSliceEnd);

      //set up movie iterators
      Graph::iterator movie_it = g.begin();
      std::advance(movie_it, workItem.movieRangeStart);
      Graph::iterator end_movie_it = g.begin();
      std::advance(end_movie_it, workItem.movieRangeEnd);

      //for each movie in the range
      for(; movie_it != end_movie_it; ++movie_it)
      {
        //get movie data
        GNode movie = *movie_it;
        Node& movie_data = g.getData(movie);

        unsigned int currentBlockSliceEndUserId = currentBlockSliceEnd + NUM_MOVIE_NODES;// + 1;

        //printf("movie %d edge_offset %d\n", movie, movie_data.edge_offset);

        //for each edge in the range
        Graph::edge_iterator edge_it = g.edge_begin(movie, Galois::MethodFlag::UNPROTECTED) + movie_data.edge_offset;
        Graph::edge_iterator edge_end = g.edge_end(movie, Galois::MethodFlag::UNPROTECTED);
        for(;edge_it != edge_end; ++edge_it, ++movie_data.edge_offset)
        {
          GNode user = g.getEdgeDst(edge_it);

          //printf("looked at user %d\n", user - NUM_MOVIE_NODES);
          //stop when you're outside the current block's user range
          if(user > currentBlockSliceEndUserId)
            break;
          //printf("okay user %d\n", user - NUM_MOVIE_NODES);

          Node& user_data = g.getData(user, Galois::MethodFlag::UNPROTECTED);
          int edge_rating = g.getEdgeData(edge_it, Galois::MethodFlag::UNPROTECTED);  

          //do gradient step
          doGradientUpdate(movie_data, user_data, edge_rating, step_size);
                                        ++movie_data.updates;

          updates++;
          //if(workItem.id == 0) printf("yoookay user %d\n", user - NUM_MOVIE_NODES);
          //if(workItem.id == 13) printf("updates %d\n", updates);
        }

        //printf("movie %d edge_offset is now %d\n", movie, movie_data.edge_offset);

        //we just looked at the last user
        if(currentBlockSliceEnd == NUM_USER_NODES)
        {
          //start back at the first edge
          movie_data.edge_offset = 0;
        }
      }

      locks[currentSliceId].unlock ();

      currentSliceId++;
      sliceUpdates++;

      //if(currentSliceId == workItem.numSlices) //hit end of slices
      if(currentBlockSliceEnd == userRangeEnd)
      {
        currentSliceId = 0;
        currentBlockSliceEnd = 0;
      }
    }

    timer.stop();
    workItem.timeTaken = timer.get_usec();
    workItem.updates = updates;
    std::cout << workItem.id << " " << workItem.updates << " " << workItem.timeTaken/1000000 << " " << conflicts << std::endl;
    //printf("Worker %d took %u to do %u updates.\n", workItem.id, workItem.timeTaken/1000000, workItem.updates);
  }

};


void runSliceMarch(Graph& g, const LearnFN* lf) {

  const unsigned threadCount = Galois::getActiveThreads ();
  unsigned numWorkItems = threadCount;
  ThreadWorkItem workItems[numWorkItems];
  unsigned int moviesPerThread = NUM_MOVIE_NODES / numWorkItems;
  unsigned int usersPerThread = NUM_USER_NODES / numWorkItems;

  unsigned int numSlices = NUM_USER_NODES / usersPerBlockSlice;

  SpinLock* locks = new SpinLock[numSlices];

  unsigned int slicesPerThread = numSlices / threadCount;
  printf("numSlices: %d slicesPerThread: %d\n", numSlices, slicesPerThread);

  //set up initial work ranges for each thread
  for(unsigned int i = 0; i < numWorkItems; i++)
  {
    ThreadWorkItem wi;
    wi.movieRangeStart = moviesPerThread * i;
    wi.userRangeStart = usersPerThread * i;
    wi.userRangeEnd = NUM_USER_NODES;

    if(i == numWorkItems - 1) //last blocks take the rest
    {
      wi.movieRangeEnd = NUM_MOVIE_NODES;
    }
    else
    {
      wi.movieRangeEnd = wi.movieRangeStart + moviesPerThread;
    }

    wi.sliceStart = slicesPerThread * i;
    wi.numSlices = numSlices;
    wi.usersPerBlockSlice = usersPerBlockSlice;

    //debug vars
    wi.id = i;
    wi.updates = 0;

    workItems[i] = wi;
  }

        double step_size = lf->step_size(1); // FIXME

  //move the edge iterators of each movie to the start of the current block
  //advances the edge iterator until it reaches the userRangeStart field of the ThreadWorkItem
  //userRangeStart isn't needed after this point
  Galois::do_all(workItems + 0, workItems + numWorkItems, advance_edge_iterators(g));
  Galois::do_all(workItems + 0, workItems + numWorkItems, sgd_march(g, locks, step_size));


  /*for(unsigned int i = 0; i < numWorkItems; i++)
    {
    ThreadWorkItem& workItem = workItems[i];
    printf("Worker %d took %lu to do %lu updates.\n", workItem.id, workItem.timeTaken/1000000, workItem.updates);
    }*/
}

typedef struct SliceInfo
{
  unsigned id;
  unsigned x;
  unsigned y;
  unsigned userStart;
  unsigned userEnd;
  unsigned movieStart;
  unsigned movieEnd;
  unsigned numMovies;
  unsigned updates;
  int* userOffsets;

  void print()
  {
    printf("id: %d x: %d y: %d userStart: %d userEnd: %d movieStart: %d movieEnd: %d updates: %d\n", 
        id, x, y, userStart, userEnd, movieStart, movieEnd, updates);
  }
} SliceInfo;


struct calculate_user_offsets
{
  Graph& g;
  SliceInfo* slices;
  unsigned moviesPerSlice, numXSlices, numYSlices, numSlices;
  calculate_user_offsets(Graph& _g, SliceInfo* _slices, unsigned _mps, unsigned _nxs, unsigned _nys) : 
    g(_g), slices(_slices), moviesPerSlice(_mps), numXSlices(_nxs), numYSlices(_nys), numSlices(_nxs * _nys) {}

  void operator()(GNode movie) const
  {
    unsigned sliceY = std::min(movie / moviesPerSlice, numYSlices - 1);
    SliceInfo* s = &slices[sliceY * numXSlices];
    
    assert(movie >= s->movieStart && movie < s->movieEnd);
    unsigned pos = movie - s->movieStart;

    /*if(movie == 15123)
    {
      printf("movie: %d val: %d numXSlices: %d sliceY: %d\n", movie, sliceY * numXSlices, numXSlices, sliceY);
    }*/
    
    //for each edge in the range
    Graph::edge_iterator edge_it = g.edge_begin(movie, Galois::MethodFlag::UNPROTECTED), 
               edge_end = g.edge_end(movie, Galois::MethodFlag::UNPROTECTED);

    for(int i = 0, offset = 0; i < numXSlices; i++, s++)
    {
      GNode user = g.getEdgeDst(edge_it);

      //in the slice's range
      if(user >= userIdToUserNode(s->userStart) && user < userIdToUserNode(s->userEnd))
      {
        s->userOffsets[pos] = offset;
        
        //move the edge_it beyond slice's range
        while(edge_it != edge_end && g.getEdgeDst(edge_it) < userIdToUserNode(s->userEnd))
        {
          ++edge_it;
          ++offset;
        }
      }
      else
      {
        s->userOffsets[pos] = -1;
      }
      
      /*if(movie == 15123)
        printf("%d ", s->userOffsets[pos]);*/
    }

    /*if(movie == 15123)
      printf("\n");*/

  }
};

struct sgd_slice_jump
{
  Graph& g;
  SpinLock *xLocks, *yLocks;
  SliceInfo* slices;
  unsigned numXSlices, numYSlices, numSlices;
  double step_size;
  sgd_slice_jump(Graph& g, SpinLock* _xLocks, SpinLock* _yLocks, SliceInfo* _slices, 
      unsigned _numXSlices, unsigned _numYSlices, double _step_size) : 
    g(g), xLocks(_xLocks), yLocks(_yLocks), slices(_slices), 
    numXSlices(_numXSlices), numYSlices(_numYSlices), numSlices(_numXSlices * _numYSlices),
    step_size(_step_size) {}
  
  //Preconditions: row and column of slice are locked
  //Postconditions: increments update count, does sgd update on each movie and user in the slice
  inline unsigned runSlice(SliceInfo* sp) const
  {
    SliceInfo& si = *sp;
    sp->updates++; //number of times slice has been updated
    unsigned edges_seen = 0;

    //set up movie iterators
    unsigned movie_num = 0;
    Graph::iterator movie_it = g.begin();
    std::advance(movie_it, si.movieStart);
    Graph::iterator end_movie_it = g.begin();
    std::advance(end_movie_it, si.movieEnd);
    
    //for each movie in the range
    for(; movie_it != end_movie_it; ++movie_it, ++movie_num)
    { 
      if(si.userOffsets[movie_num] < 0)
        continue;

      //get movie data
      GNode movie = *movie_it;
      Node& movie_data = g.getData(movie);

      unsigned int currentBlockSliceEndUserId = si.userEnd + NUM_MOVIE_NODES;

      //for each edge in the range
      Graph::edge_iterator edge_it = g.edge_begin(movie, Galois::MethodFlag::UNPROTECTED) + si.userOffsets[movie_num];
      Graph::edge_iterator edge_end = g.edge_end(movie, Galois::MethodFlag::UNPROTECTED);
      for(;edge_it != edge_end; ++edge_it, ++movie_data.edge_offset)
      {
        GNode user = g.getEdgeDst(edge_it);

        //stop when you're outside the current block's user range
        if(user > currentBlockSliceEndUserId)
          break;

        Node& user_data = g.getData(user, Galois::MethodFlag::UNPROTECTED);
        int edge_rating = g.getEdgeData(edge_it, Galois::MethodFlag::UNPROTECTED);  

        //do gradient step
        doGradientUpdate(movie_data, user_data, edge_rating, step_size);

        ++edges_seen;
      }
    }
    
    //printf("Slice (%d,%d) Edges Seen: %d\n", si.x, si.y, edges_seen);
    return edges_seen;
  }
  
  //determines the next slice to work on
  //returns: slice id to work on, x and y locks are held on the slice
  inline int getNextSlice(SliceInfo* sp) const
  {
    bool foundWork = false;
    unsigned workSearchTries = 0;
    unsigned nextSliceId = sp->id;
    while(foundWork || workSearchTries < 2 * numSlices)
    {
      workSearchTries++;

      nextSliceId++;
      if(nextSliceId == numSlices) nextSliceId = 0; //wrap around

      SliceInfo& nextSlice = slices[nextSliceId];

      if(nextSlice.updates < MAX_MOVIE_UPDATES && xLocks[nextSlice.x].try_lock())
      {
        if(yLocks[nextSlice.y].try_lock())
        {
          foundWork = true;
          break; //break while holding x and y locks
        }
        else
        {
          xLocks[nextSlice.x].unlock();
        }
      }
    }

    /*if(foundWork)
      printf("Found work after %d tries\n", workSearchTries);
    else
      printf("Didn't find work after %d tries\n", workSearchTries);
    */

    return foundWork ? nextSliceId : -1;
  }

  void operator()(SliceInfo* startSlice) const
  {
                Galois::Statistic edgesVisited("EdgesVisited");

    Galois::Timer timer;
    timer.start();

    SliceInfo* sp = startSlice;
    bool go = false;
    unsigned tot_updates = 0;
    unsigned slices_updated = 0;

    do {
      //x and y are taken for sp at this point
      int edges_visited = runSlice(sp);
      tot_updates += edges_visited;
      slices_updated++;

      xLocks[sp->x].unlock();
      yLocks[sp->y].unlock();
  
      int nextWorkId = getNextSlice(sp);
      if(nextWorkId >= 0)
      {
        go = true; //keep going if we found some work
        sp = &slices[nextWorkId]; //set next work
      }
      else
        go = false;

    } while(go);

    timer.stop();
    printf("Slice: (%d, %d) Edges: %d Slices: %d Time: %f\n", startSlice->x, startSlice->y, tot_updates, slices_updated, timer.get_usec()/1000000.0);
                edgesVisited += tot_updates;
  }
};


void runSliceJump(Graph& g, const LearnFN* lf)
{
  //set up parameters
  const unsigned threadCount = Galois::getActiveThreads();
  //threadCount + 1 so free slices are always available
  const unsigned numXSlices = std::max(NUM_USER_NODES/ usersPerBlockSlice, threadCount + 1);
  const unsigned numYSlices  = std::max(NUM_MOVIE_NODES/ moviesPerBlockSlice, threadCount + 1);
  const unsigned numSlices = numXSlices * numYSlices;
  const unsigned moviesPerSlice = NUM_MOVIE_NODES / numYSlices;
  const unsigned usersPerSlice = NUM_USER_NODES / numXSlices;

  SpinLock* xLocks = new SpinLock[numXSlices];
  SpinLock* yLocks = new SpinLock[numYSlices];
  
  printf("numSlices: %d numXSlices %d numYSlices %d\n", numSlices, numXSlices, numYSlices);
  
  //initialize slice infos
  SliceInfo* slices = new SliceInfo[numSlices];
  for(unsigned i = 0; i < numSlices; i++)
  {
    SliceInfo& si = slices[i];
    si.id = i;
    si.x = i % numXSlices;
    si.y = i / numXSlices;
    si.userStart = si.x * usersPerSlice;
    si.userEnd = si.x == numXSlices - 1 ? NUM_USER_NODES : (si.x + 1) * usersPerSlice;
    si.movieStart = si.y * moviesPerSlice;
    si.movieEnd = si.y == numYSlices -1 ? NUM_MOVIE_NODES : (si.y + 1) * moviesPerSlice;
    si.updates = 0;
    
    si.numMovies = si.movieEnd - si.movieStart;
    si.userOffsets = new int[si.numMovies];
  }

  //generate indexes for user ids into graph
  Galois::do_all(g.begin(), g.begin() + NUM_MOVIE_NODES, 
      calculate_user_offsets(g, slices, moviesPerSlice, numXSlices, numYSlices));
  
  //stagger the starting slices for each thread
  SliceInfo** startSlices = new SliceInfo*[threadCount];
  float xSliceGap = numXSlices / (float) threadCount;
  float ySliceGap = numYSlices / (float) threadCount;
  //printf("xgap: %f ygap: %f\n", xSliceGap, ySliceGap);
  for(unsigned i = 0; i < threadCount; i++)
  {
    unsigned xSlice = (unsigned) (xSliceGap * i);
    unsigned ySlice = (unsigned) (ySliceGap * i);
    xLocks[xSlice].lock();
    yLocks[ySlice].lock();
    //printf("slices: %d %d\n", xSlice, ySlice);
    startSlices[i] = &slices[ xSlice + ySlice * numXSlices];
  }
  
  double step_size = lf->step_size(1); //FIXME
  Galois::do_all(&startSlices[0], &startSlices[threadCount], 
      sgd_slice_jump(g, xLocks, yLocks, slices, numXSlices, numYSlices, step_size));
}

static double genRand () {
  // generate a random double in (-1,1)
  return 2.0 * ((double)std::rand () / (double)RAND_MAX) - 1.0;
}

// Initializes latent vector and id for each node
std::pair<unsigned int, unsigned int> initializeGraphData(Graph& g)
{
  // unsigned int seed = 42;
  // std::default_random_engine eng(seed);
  // std::uniform_real_distribution<double> random_lv_value(0, 0.1);
  const unsigned SEED = 4562727;
  std::srand (SEED);

  unsigned int numMovieNodes = 0;
        unsigned int numUserNodes = 0;
  unsigned int numRatings = 0;

  //for all movie and user nodes in the graph
  for (Graph::iterator i = g.begin(), end = g.end(); i != end; ++i) {
    GNode gnode = *i;
    Node& data = g.getData(gnode);

    data.updates = 0;

    //fill latent vectors with random values
    for(int i = 0; i < LATENT_VECTOR_SIZE; i++)
      data.latent_vector[i] = genRand();

    //count number of movies we've seen; only movies nodes have edges
    unsigned int num_edges = 
      g.edge_end(gnode, Galois::MethodFlag::UNPROTECTED) - g.edge_begin(gnode, Galois::MethodFlag::UNPROTECTED);
    numRatings += num_edges;
    if(num_edges > 0)
                  numMovieNodes++;
                else
                  numUserNodes++;

    data.edge_offset = 0;
  }

  NUM_RATINGS = numRatings;

  return std::make_pair(numMovieNodes, numUserNodes);
}

int main(int argc, char** argv) { 

  LonestarStart (argc, argv, name, desc, url);
  Galois::StatManager statManager;

  //allocate local computation graph
  Graph g;

  //read structure of graph & edge weights; nodes not initialized
  Galois::Graph::readGraph(g, inputFile);

  //fill each node's id & initialize the latent vectors
  unsigned int numMovieNodes, numUserNodes;
        std::tie(numMovieNodes, numUserNodes) = initializeGraphData(g);

  std::cout << "Input initialized, num users = " << numUserNodes 
    << ", num movies = " << numMovieNodes << std::endl;

  NUM_MOVIE_NODES = numMovieNodes;
  NUM_USER_NODES = numUserNodes;

        if (shiftFactor != 0)
          for (auto ii = g.begin(), ee = g.end(); ii != ee; ++ii)
            for (auto eii = g.edge_begin(*ii), eee = g.edge_end(*ii); eii != eee; ++eii)
              g.getEdgeData(eii) -= shiftFactor;
  
  Galois::StatTimer timer;
  timer.start();

        std::unique_ptr<LearnFN> lf;
        switch (learn) {
        case Intel:
          lf.reset(new IntelLearnFN);
          break;
        case Purdue:
          lf.reset(new PurdueLearnFN);
          break;
        case Bottou:
          lf.reset(new BottouLearnFN);
          break;
        case Inv:
          lf.reset(new InvLearnFN);
          break;
        }

  switch (algo) {
        case Algo::nodeMovie:
          sgd_node_movie::go(g, numMovieNodes, numUserNodes, lf.get());
          break;
        case Algo::nodeMoviePri:
          sgd_node_movie_pri::go(g, numMovieNodes, numUserNodes, lf.get());
          break;
        case Algo::edgeMovie:
          sgd_edge_movie::go(g, numMovieNodes, numUserNodes, lf.get());
          break;
        case Algo::block:
          runBlockSlices<sgd_block>(g, lf.get());
          break;
        case Algo::blockAndSliceUsers:
          runBlockSlices<sgd_block_users>(g, lf.get());
          break;
        case Algo::blockAndSliceBoth:
          runBlockSlices<sgd_block_users_movies>(g, lf.get());
          break;
        case Algo::sliceMarch:
          runSliceMarch(g, lf.get());
          break;
    case Algo::sliceJump:
      runSliceJump(g, lf.get());
      break;
        }

  timer.stop();

  //verify (g);

  std::cout << "SUMMARY Movies " << numMovieNodes << 
    " Users " << numUserNodes <<
    " Ratings " << g.sizeEdges() << 
    " usersPerBlockSlice " << usersPerBlockSlice << 
    " moviesPerBlockSlice " << moviesPerBlockSlice << 
    " Time " << timer.get()/1000.0 << std::endl;

  return 0;
}
