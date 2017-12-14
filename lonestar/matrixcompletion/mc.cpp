/** Matrix completion through Stochastic Gradient Descent -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2017, The University of Texas at Austin. All rights reserved.
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
 * Stochastic gradient descent for matrix factorization, implemented with 
 * Galois.
 *
 * @author Prad Nelluru <pradn@cs.utexas.edu>
 * @author Loc Hoang <lhoang@utexas.edu> (Some utility code/Code cleanup)
 */

#include "galois/Galois.h"
#include "galois/graphs/LCGraph.h"
#include "llvm/Support/CommandLine.h"
#include "Lonestar/BoilerPlate.h"
#include <cmath>
#include <random>

static const char* const name = "Matrix Completion";
static const char* const desc = "Computes Matrix Decomposition using Stochastic "
                                "Gradient Descent";
static const char* const url = 0;

// This isn't a command line option because this number needs to be known
// to define the Node data type
static const unsigned LATENT_VECTOR_SIZE = 20;

static const double MINVAL = -1e+100;
static const double MAXVAL = 1e+100;

enum Algo {
  nodeMovie,
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

////////////////////////////////////////////////////////////////////////////////
// Command line parameters
////////////////////////////////////////////////////////////////////////////////

namespace cll = llvm::cl;

static cll::opt<std::string> inputFile(cll::Positional, 
                                       cll::desc("<input file>"), 
                                       cll::Required);
static cll::opt<unsigned> usersPerBlockSlice("usersPerBlk", 
                                             cll::desc("Users per block slice "
                                                       "(default 2048)"), 
                                             cll::init(2048));
static cll::opt<unsigned> moviesPerBlockSlice("moviesPerBlk", 
                                              cll::desc("Movies per block "
                                                        "slice (default 100)"), 
                                              cll::init(350));
static cll::opt<bool> verifyPerIter("verifyPerIter", 
                                    cll::desc("Prints RMS every iteration"), 
                                    cll::init(false));
static cll::opt<int> shiftFactor("shiftRating", 
                                 cll::desc("Shift ratings down by some constant"),
                                 cll::init(0));

// Regarding algorithm termination
static cll::opt<unsigned> maxUpdates("maxUpdates", 
                                     cll::desc("Max number of times to update "
                                               "latent vectors (default 100)"), 
                                     cll::init(15));
static cll::opt<double> errorThreshold("errorThreshold", 
                                       cll::desc("Algo terminates when root "
                                                 "mean square error is less "
                                                 "than this number (default 1)"), 
                                       cll::init(1.0));

// Selection of algo variant + learning function
static cll::opt<Algo> algo("algo", cll::desc("Algorithm variant:"),
                           cll::values(
                             clEnumVal(nodeMovie, "Node by Movies"),
                             clEnumVal(edgeMovie, "Edge by Movies"),
                             clEnumVal(block, "Block by Users and Movies"),
                             clEnumVal(blockAndSliceUsers, 
                                       "Block by Users and Movies, Slice by "
                                       "Users"),
                             clEnumVal(blockAndSliceBoth, 
                                       "Block by Users and Movies, Slice by "
                                       "Users and Movies (default)"),
                             clEnumVal(sliceMarch, "Marching Slices version"),
                             clEnumVal(sliceJump, "Jumping Slices version"),
                             clEnumValEnd
                           ), 
                           cll::init(blockAndSliceBoth));
static cll::opt<Learn> learn("lf", cll::desc("Learning function:"),
                             cll::values(
                               clEnumVal(Intel, "Intel (default)"),
                               clEnumVal(Purdue, "Purdue"),
                               clEnumVal(Bottou, "Bottou"),
                               clEnumVal(Inv, "Simple Inverse"),
                               clEnumValEnd
                             ), 
                             cll::init(Intel));

// Learning params
static cll::opt<double> learningRate("learningRate",
                                     cll::desc("Parameter to adjust rate of "
                                               "learning by way of step size"
                                               " (default 0.001)"),
                                     cll::init(0.001));
static cll::opt<double> intelDecayRate("intelDecay",
                                       cll::desc("Parameter to adjust rate of "
                                                 "step size decay for Intel "
                                                 "learning; higher = slower "
                                                 "decay (default 0.9)"),
                                       cll::init(0.9));
static cll::opt<double> purdueDecayRate("purdueDecay",
                                       cll::desc("Parameter to adjust rate of "
                                                 "step size decay for Purdue "
                                                 "learning; higher = faster "
                                                 "decay (default 0.1)"),
                                       cll::init(0.1));
static cll::opt<double> lambda("lambda",
                              cll::desc("Lambda learning parameter for "
                                        "gradient change and Bottou "
                                        "(default 0.001)"),
                              cll::init(0.001));
static cll::opt<double> bottouInit("bottouInit",
                                   cll::desc("Bottou learning function init "
                                             "value (default 0.1)"),
                                   cll::init(0.1));

////////////////////////////////////////////////////////////////////////////////
// Declaration of strutures, types, and variables
////////////////////////////////////////////////////////////////////////////////

/**
 * Every graph node is represented by this structure.
 */
struct Node {
  // latent vector to be learned
  double latent_vector[LATENT_VECTOR_SIZE]; 
  // number of updates made to this node (only used by movie nodes)
  unsigned int updates; 
  // if a movie's update is interrupted, where to start when resuming
  unsigned int edge_offset;

  /**
   * Print the latent vector.
   *
   * @param os output stream to print the vector to
   */
  void dump(std::ostream& os) {
    os << "{" << latent_vector[0];

    for (unsigned i = 1; i < LATENT_VECTOR_SIZE; ++i) os << ", " << latent_vector[i];

    os << "}";
  }
};

using Graph = typename galois::graphs::LC_CSR_Graph<Node, int>
                         ::with_numa_alloc<true>::type
                         ::with_no_lockable<false>::type;
using GNode = Graph::GraphNode;

/**
 * Structure that holds the information needed for a thread do to SGD.
 */
struct ThreadWorkItem {
  unsigned int movieRangeStart;
  unsigned int movieRangeEnd;
  unsigned int userRangeStart;
  unsigned int userRangeEnd;
  unsigned int usersPerBlockSlice;
  unsigned int moviesPerBlockSlice;

  unsigned int sliceStart; // only used in march variation
  unsigned int numSlices; // only used in march variation

  // debug variables
  unsigned int id;
  unsigned int updates;
  double timeTaken;
};

unsigned int NUM_MOVIE_NODES = 0;
unsigned int NUM_USER_NODES = 0;
unsigned int NUM_RATINGS = 0;

////////////////////////////////////////////////////////////////////////////////
// Utility funcions for use during execution
////////////////////////////////////////////////////////////////////////////////

/** 
 * Generate a random double in (-1, 1)
 *
 * @returns Random double in (-1, 1)
 */
static double genRand() {
  return 2.0 * ((double)std::rand() / (double)RAND_MAX) - 1.0;
}

/** 
 * Initializes latent vector and id for each node.
 *
 * @param g Graph to initialize
 * @returns pair with number of movies and number of users in the graph
 */
std::pair<unsigned int, unsigned int> initializeGraphData(Graph& g) {
  const unsigned SEED = 4562727;
  std::srand(SEED);

  unsigned int numMovieNodes = 0;
  unsigned int numUserNodes = 0;
  unsigned int numRatings = 0;

  // loop through all movie and user nodes
  for (Graph::iterator i = g.begin(), end = g.end(); i != end; ++i) {
    GNode gnode = *i;
    Node& data = g.getData(gnode);

    // fill latent vectors with random values
    for (unsigned i = 0; i < LATENT_VECTOR_SIZE; i++)
      data.latent_vector[i] = genRand();

    // edges are ratings
    unsigned int num_edges = 
      g.edge_end(gnode, galois::MethodFlag::UNPROTECTED) - 
      g.edge_begin(gnode, galois::MethodFlag::UNPROTECTED);

    numRatings += num_edges;

    // only movies should have edges
    if (num_edges > 0) {
      numMovieNodes++;
    } else {
      numUserNodes++;
    }

    data.updates = 0;
    data.edge_offset = 0;
  }

  NUM_RATINGS = numRatings;

  return std::make_pair(numMovieNodes, numUserNodes);
}

// possibly over-typed
/**
 * Gets dot product of latent vectors of 2 nodes.
 *
 * @param movieData movie data node
 * @param userData user data node
 * @returns Dot product of latent vectors in movieData and userData
 */
double vectorDot(const Node& movieData, const Node& userData) {
  // Could just specify restrict on parameters since vector is built in
  const double* __restrict__ movieLatent = movieData.latent_vector;
  const double* __restrict__ userLatent = userData.latent_vector;

  double dotProduct = 0.0;
  for (unsigned i = 0; i < LATENT_VECTOR_SIZE; ++i) {
    dotProduct += userLatent[i] * movieLatent[i];
  }

  assert(std::isnormal(dotProduct));
  return dotProduct;
}

/**
 * Get the prediction of a rating based on latent vectors on the movie
 * and the user.
 * 
 * @param movieData movie node data
 * @param userData user node data
 *
 * @returns Prediction of what the edge weight is.
 */
double calcPrediction(const Node& movieData, const Node& userData) {
  double pred = vectorDot(movieData, userData);
  double p = pred;

  pred = std::min(MAXVAL, pred);
  pred = std::max(MINVAL, pred);

  if (p != pred) {
    galois::gPrint("calcPrediction: Clamped ", p, " to ", pred, "\n");
  }

  return pred;
}

/**
 * Does a gradient update of 2 latent vectors on 2 nodes in order to reduce
 * the error of the prediction of the edge data.
 * 
 * @param movieData movie node data
 * @param userData user node data
 * @param edgeRating The actual weight of the edge (i.e. ground truth)
 * @param stepSize Factor determining how much to change the latent vector by
 * @returns Error of prediction before gradient update.
 */
double doGradientUpdate(Node& movieData, Node& userData, int edgeRating, 
                        double stepSize) {
  double* __restrict__ movieLatent = movieData.latent_vector;
  double* __restrict__ userLatent = userData.latent_vector;
  
  // calculate error
  double oldDP = vectorDot(movieData, userData);
  double curError = edgeRating - oldDP;
  assert(curError < 1000 && curError > -1000);
  
  // take gradient step
  for (unsigned i = 0; i < LATENT_VECTOR_SIZE; i++) {
    double prevMovieVal = movieLatent[i];
    double prevUserVal = userLatent[i];

    movieLatent[i] += stepSize * (curError * prevUserVal - 
                                  lambda * prevMovieVal);
    assert(std::isnormal(movieLatent[i]));

    userLatent[i] += stepSize * (curError * prevMovieVal - 
                                 lambda * prevUserVal);
    assert(std::isnormal(userLatent[i]));
  }

  return curError;
}

/**
 * Calculate root mean square error of edge weight predictions.
 *
 * @param g Graph with nodes that have latent vectors
 * @returns root mean squared error of predictions of edge weights
 */
double rootMeanSquaredError(Graph& g) {
  galois::GAccumulator<double> rms;

  galois::do_all(
    galois::iterate(g),
    [&g, &rms] (GNode n) {
      for (auto e : g.edges(n, galois::MethodFlag::UNPROTECTED)) {
        GNode m = g.getEdgeDst(e);

        double pred = 
          calcPrediction(g.getData(n, galois::MethodFlag::UNPROTECTED), 
                         g.getData(m, galois::MethodFlag::UNPROTECTED));

        if (!std::isnormal(pred)) {
          galois::gWarn("not normal prediction warning");
        }

        double rating = g.getEdgeData(e, galois::MethodFlag::UNPROTECTED);

        if (pred > 100.0 || pred < -100.0) {
          galois::gWarn("Big difference: prediction ", pred, " should be ", 
                        rating);
        }

        rms += ((pred - rating) * (pred - rating));
      }
    }
  );

  double rmsSum = rms.reduce();
  double rmsMean = rmsSum / g.sizeEdges();
  return sqrt(rmsMean);
}

/**
 * Computes root mean square error of predictions and prints it.
 * Assumes only movie nodes have edges
 *
 * @param g Graph with nodes that have latent vectors
 */
void verify(Graph& g) {
  double totalRMS = rootMeanSquaredError(g);
  galois::gPrint("RMSE Total: ", totalRMS, "\n");
}

/**
 * Given a user id, return the node id of the user in the graph.
 *
 * @param userId ID of user to get the graph node of
 * @returns Graph ID of the user
 */
unsigned int userIdToUserNode(unsigned int userId) {
  return userId + NUM_MOVIE_NODES;
}

/**
 * Find the initial edge offsets for each thread work item (i.e. each movie
 * sets its edge offset to point to the first user in its assigned
 * user range.
 */
struct AdvanceEdgeOffsets {
  Graph& g;

  AdvanceEdgeOffsets(Graph& g) : g(g) {}

  void operator()(ThreadWorkItem& workItem) const {
    Graph::iterator startMovie = g.begin();
    std::advance(startMovie, workItem.movieRangeStart);

    Graph::iterator endMovie = g.begin();
    std::advance(endMovie, workItem.movieRangeEnd);

    // go through movies and set an edge offset to the first user that
    // the movie should start iteration from
    for (; startMovie != endMovie; ++startMovie) {
      GNode movie = *startMovie;
      Node& movieData = g.getData(movie, galois::MethodFlag::UNPROTECTED);

      Graph::edge_iterator curEdge = 
        g.edge_begin(movie, galois::MethodFlag::UNPROTECTED) + 
        movieData.edge_offset;

      Graph::edge_iterator endEdge = 
        g.edge_end(movie, galois::MethodFlag::UNPROTECTED);

      for (; curEdge != endEdge; ++curEdge, ++movieData.edge_offset) {
        GNode userNode = g.getEdgeDst(curEdge);
        // stop when you're in the acceptable user range to start from
        // based on what the work item says the user to start from is
        if (userNode >= userIdToUserNode(workItem.userRangeStart)) break;
      }
    }
  }
};

// utility function to learn about a graph input
void count_ratings(Graph& g) {
  const unsigned threadCount = galois::getActiveThreads ();

  std::vector<unsigned long> ratings_per_user(NUM_USER_NODES);
  std::vector<unsigned long> ratings_per_movie(NUM_MOVIE_NODES);

  Graph::iterator movie_it = g.begin();
  Graph::iterator end_movie_it = g.end();

  for (; movie_it != end_movie_it; ++movie_it) {
    //get movie data
    GNode movie = *movie_it;
    Node& movie_data = g.getData(movie);

    //for each edge in the range
    Graph::edge_iterator edge_it = g.edge_begin(movie, galois::MethodFlag::UNPROTECTED) + movie_data.edge_offset;
    Graph::edge_iterator edge_end = g.edge_end(movie, galois::MethodFlag::UNPROTECTED);
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

////////////////////////////////////////////////////////////////////////////////
// Learning Functions: Specify learning step size that is dependent on round
////////////////////////////////////////////////////////////////////////////////

struct PurdueLearnFN {
  double step_size(unsigned int round) const {
    return (learningRate * 1.5) / (1.0 + purdueDecayRate * pow(round + 1, 1.5));
  }
};

struct IntelLearnFN {
  double step_size(unsigned int round) const {
    return learningRate * pow(intelDecayRate, round);
  }
};

struct BottouLearnFN {
  double step_size(unsigned int round) const {
    return bottouInit / (1 + bottouInit * lambda * round);
  }
};

struct InvLearnFN {
  double step_size(unsigned int round) const {
    return (double)1 / (double)(round + 1);
  }
};


////////////////////////////////////////////////////////////////////////////////
// Matrix completion algorithm variants
////////////////////////////////////////////////////////////////////////////////

/**
 * SGD simple by-movie node-based: each thread does updates for a subset
 * of the nodes. getData on user nodes should have locked access.
 *
 * User nodes should not have edges.
 */
void SGDNodeMovie(Graph& g, const LearnFN* lf) {
  unsigned numIterations = 0;
  double latestError = rootMeanSquaredError(g);

  if (verifyPerIter) galois::gPrint("Initial RMS: ", latestError, "\n");

  while (numIterations < maxUpdates && latestError > errorThreshold) {
    double step_size = lf->step_size(numIterations);
    galois::gDebug("Step Size: ", step_size);

    galois::for_each(
      galois::iterate(g), 
      [&] (GNode node, auto& ctx) {
        for (auto ii : g.edges(node)) {
          doGradientUpdate(g.getData(node, galois::MethodFlag::UNPROTECTED), 
                           g.getData(g.getEdgeDst(ii)), g.getEdgeData(ii), 
                           step_size);
        }
      },
      galois::wl<galois::worklists::dChunkedFIFO<8>>(),
      galois::no_pushes(),
      galois::loopname("SGDNodeMovie")
    );

    latestError = rootMeanSquaredError(g);
    if (verifyPerIter) galois::gPrint("RMS: ", latestError, "\n");
    numIterations++;
  }
}

//priority by-movie node-based
// TODO this implementation seems non-sensical (why are you pushing
// user nodes to the worklist when they have no edges?)
/** 
 * Simple by-edge grouped by movie (only one edge per movie on the WL at any 
 * time)
 */
void SGDEdgeMovie(Graph& g, const LearnFN* lf) {
  galois::InsertBag<GNode> Movies;

  // get only nodes with edges
  galois::do_all(
    galois::iterate(g),
    [&] (GNode n) {
      if (g.edge_begin(n) != g.edge_end(n)) Movies.push_back(n);
    }
  );

  unsigned numIterations = 0;
  double latestError = rootMeanSquaredError(g);
  if (verifyPerIter) galois::gPrint("Initial RMS: ", latestError, "\n");

  while (numIterations < maxUpdates && latestError > errorThreshold) {
    double step_size = lf->step_size(numIterations);
    galois::gDebug("Step Size: ", step_size);

    galois::for_each(
      galois::iterate(Movies),
      [&] (GNode node, auto& ctx) {
        auto ii = g.edge_begin(node, galois::MethodFlag::UNPROTECTED); 
        auto ee = g.edge_end(node, galois::MethodFlag::UNPROTECTED);

        if (ii == ee) return;

        auto& nd = g.getData(node, galois::MethodFlag::UNPROTECTED);
        std::advance(ii, nd.edge_offset);
        auto& no = g.getData(g.getEdgeDst(ii));
        doGradientUpdate(nd, no, g.getEdgeData(ii), step_size);

        ++nd.edge_offset;
        ++ii;

        if (ii == ee) {
          nd.edge_offset = 0; return;
        } else {
          ctx.push(node);
        }
      },
      galois::wl<galois::worklists::dChunkedLIFO<8>>(),
      galois::loopname("SGDEdgeMovie")
    );

    latestError = rootMeanSquaredError(g);
    if (verifyPerIter) galois::gPrint("RMS: ", latestError, "\n");
    numIterations++;
  }
}

/**
 * Each thread is assigned an immutable set of movies to work on. The
 * users each thread works on every time this is called are cycled among
 * threads by the caller of this functor.
 */
struct SGDBlock {
  Graph& g;
  double stepSize;
  SGDBlock(Graph& g, double ss) : g(g), stepSize(ss) {}

  void operator()(ThreadWorkItem& workItem) const {
    galois::Timer timer;
    timer.start();

    int updates = 0;

    // set up movie iterators
    Graph::iterator currentMovie = g.begin();
    std::advance(currentMovie, workItem.movieRangeStart);
    Graph::iterator endMovie = g.begin();
    std::advance(endMovie, workItem.movieRangeEnd);

    // go through each movie and work on this work item's assigned users
    for (; currentMovie != endMovie; ++currentMovie) {
      GNode movie = *currentMovie;
      Node& movieData = g.getData(movie, galois::MethodFlag::UNPROTECTED);

      auto currentEdge = g.edge_begin(movie, galois::MethodFlag::UNPROTECTED) + 
                         movieData.edge_offset;
      auto edgeEnd = g.edge_end(movie, galois::MethodFlag::UNPROTECTED);

      for (; currentEdge != edgeEnd; ++currentEdge, ++movieData.edge_offset) {
        GNode user = g.getEdgeDst(currentEdge);

        // stop when you're outside the current block's user range
        if (user >= workItem.userRangeEnd) break;

        Node& userData = g.getData(user, galois::MethodFlag::UNPROTECTED);
        int edgeRating = g.getEdgeData(currentEdge, 
                                       galois::MethodFlag::UNPROTECTED);  

        doGradientUpdate(movieData, userData, edgeRating, stepSize);

        ++movieData.updates;
        updates++;
      }

      // we just looked at the last user; loop back to first edge
      if (workItem.userRangeEnd == NUM_USER_NODES) {
        movieData.edge_offset = 0;
      }
    }

    timer.stop();

    workItem.timeTaken = timer.get_usec();
    workItem.updates = updates;
  }
};

/**
 * Each thread works on a disjoint portion of users/movies (cycle users 
 * among each other), but users are blocked within a thread.
 */
struct SGDBlockUsers {
  Graph& g;
  double stepSize;
  SGDBlockUsers(Graph& g, double ss) : g(g), stepSize(ss) {}

  void operator()(ThreadWorkItem& workItem) const {
    galois::Timer timer;
    timer.start();

    int updates = 0;

    unsigned int currentBlockSliceEnd = workItem.userRangeStart;

    while (currentBlockSliceEnd < workItem.userRangeEnd) {
      currentBlockSliceEnd += workItem.usersPerBlockSlice;
      currentBlockSliceEnd = std::min(currentBlockSliceEnd, 
                                      workItem.userRangeEnd);

      Graph::iterator currentMovie = g.begin();
      std::advance(currentMovie, workItem.movieRangeStart);
      Graph::iterator endMovie = g.begin();
      std::advance(endMovie, workItem.movieRangeEnd);

      for (; currentMovie != endMovie; ++currentMovie) {
        GNode movie = *currentMovie;
        Node& movieData = g.getData(movie, galois::MethodFlag::UNPROTECTED);

        unsigned int currentBlockSliceEndUserId = currentBlockSliceEnd + 
                                                  NUM_MOVIE_NODES;

        Graph::edge_iterator currentEdge = 
            g.edge_begin(movie, galois::MethodFlag::UNPROTECTED) + 
            movieData.edge_offset;
        Graph::edge_iterator endEdge = 
            g.edge_end(movie, galois::MethodFlag::UNPROTECTED);

        for (; currentEdge != endEdge; ++currentEdge, ++movieData.edge_offset) {
          GNode user = g.getEdgeDst(currentEdge);

          // stop when you're outside the current block's user range
          if (user >= currentBlockSliceEndUserId) break;

          Node& userData = g.getData(user, galois::MethodFlag::UNPROTECTED);
          int edgeRating = g.getEdgeData(currentEdge, 
                                         galois::MethodFlag::UNPROTECTED);  

          doGradientUpdate(movieData, userData, edgeRating, stepSize);

          ++movieData.updates;
          updates++;
        }

        // we just looked at the last user; loop back to beginning
        if (currentBlockSliceEnd == NUM_USER_NODES) {
          movieData.edge_offset = 0;
        }
      }
    }

    timer.stop();

    workItem.timeTaken = timer.get_usec();
    workItem.updates = updates;
  }

};

/**
 * Do one round of latent vector updates: users and movies are done in blocks
 * with threads cycling the blocks among each other.
 */ 
struct SGDBlockUsersMovies {
  Graph& g;
  double stepSize;
  SGDBlockUsersMovies(Graph& g, double ss) : g(g), stepSize(ss) {}

  void operator()(ThreadWorkItem& workItem) const {
    galois::Timer opTimer;

    opTimer.start();
    int updates = 0;

    unsigned int currentUserSliceEnd = workItem.userRangeStart;

    while (currentUserSliceEnd < workItem.userRangeEnd) {
      currentUserSliceEnd += workItem.usersPerBlockSlice;
      currentUserSliceEnd = std::min(currentUserSliceEnd, 
                                     workItem.userRangeEnd);

      unsigned int currentMovieSliceEnd = workItem.movieRangeStart;

      while (currentMovieSliceEnd < workItem.movieRangeEnd) {
        Graph::iterator currentMovie = g.begin();
        std::advance(currentMovie, currentMovieSliceEnd);

        currentMovieSliceEnd += workItem.moviesPerBlockSlice;
        currentMovieSliceEnd = std::min(currentMovieSliceEnd, 
                                        workItem.movieRangeEnd);

        Graph::iterator endMovie = g.begin();
        std::advance(endMovie, currentMovieSliceEnd);

        // for each movie in the range
        for (; currentMovie != endMovie; ++currentMovie) {
          GNode movie = *currentMovie;
          Node& movieData = g.getData(movie, galois::MethodFlag::UNPROTECTED);

          unsigned int currentUserSliceEndUserId = currentUserSliceEnd + 
                                                   NUM_MOVIE_NODES;

          Graph::edge_iterator currentEdge = 
            g.edge_begin(movie, galois::MethodFlag::UNPROTECTED) + 
            movieData.edge_offset;

          Graph::edge_iterator edgeEnd = 
            g.edge_end(movie, galois::MethodFlag::UNPROTECTED);

          // note edge_offset is incremented as well since you will work on that
          // next chunk of users next
          for (; 
               currentEdge != edgeEnd; 
               ++currentEdge, ++movieData.edge_offset) {
            GNode user = g.getEdgeDst(currentEdge);

            // stop when you're outside the current block's user range
            if (user >= currentUserSliceEndUserId) break;

            Node& userData = g.getData(user, galois::MethodFlag::UNPROTECTED);
            int edgeRating = g.getEdgeData(currentEdge, 
                                           galois::MethodFlag::UNPROTECTED);  

            doGradientUpdate(movieData, userData, edgeRating, stepSize);

            movieData.updates++;
            updates++;
          }

          // we just looked at the last user; loop back to the first edge/users
          if (currentUserSliceEnd == NUM_USER_NODES) {
            movieData.edge_offset = 0;
          }
        }
      }
    }

    opTimer.stop();

    workItem.timeTaken = opTimer.get_usec();
    workItem.updates = updates;
  }

};


/**
 * Common code for running blocked variants of SGD (movies are statically
 * assigned to a thread, and user nodes are cycled among the threads).
 */
template<typename BlockFn>
void runBlockSlices(Graph& g, const LearnFN* lf) {
  // Tune a const multiplication factor
  const unsigned numWorkItems = C * galois::getActiveThreads();

  ThreadWorkItem workItems[numWorkItems];

  unsigned int moviesPerThread = NUM_MOVIE_NODES / numWorkItems;
  unsigned int usersPerThread = NUM_USER_NODES / numWorkItems;

  unsigned int userRangeStartPoints[numWorkItems];
  unsigned int userRangeEndPoints[numWorkItems];

  // set up initial work ranges for each thread
  for (unsigned int i = 0; i < numWorkItems; i++) {
    ThreadWorkItem wi;
    // setup start points
    wi.movieRangeStart = moviesPerThread * i;
    wi.userRangeStart = usersPerThread * i;

    // stored for the AdvanceEdgeOffsets step
    userRangeStartPoints[i] = wi.userRangeStart;

    // setup the end points
    if (i != numWorkItems - 1) {
      wi.movieRangeEnd = (i + 1) * moviesPerThread;
      wi.userRangeEnd = (i + 1) * usersPerThread;
    } else {
      // this is the last block; it takes the rest
      wi.movieRangeEnd = NUM_MOVIE_NODES;
      wi.userRangeEnd = NUM_USER_NODES;
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

  // Advances the edge iterator until it reaches the userRangeStart field of 
  // the ThreadWorkItem; save the offset to the movie node data
  galois::do_all(
    galois::iterate(workItems + 0, workItems + numWorkItems), 
    AdvanceEdgeOffsets(g)
  );

  unsigned numIterations = 0;
  double latestError = rootMeanSquaredError(g);

  if (verifyPerIter) galois::gPrint("Initial RMS: ", latestError, "\n");

  while (numIterations < maxUpdates && latestError > errorThreshold) {
    // work on the current blocks, move the block a thread works on to the right
    for (unsigned int j = 0; j < numWorkItems; j++) {  
      // Each thread works on 1 work item
      galois::do_all(
        galois::iterate(workItems + 0, workItems + numWorkItems),
        BlockFn(g, lf->step_size(numIterations))
      );

      // move each thread's user assingment one block to the right
      // (i.e. for the same movie nodes, look at the next range of user nodes)
      for (unsigned int k = 0; k < numWorkItems; k++) {
        ThreadWorkItem& wi = workItems[k];
        unsigned int nextColumn = (j + 1 + k) % numWorkItems;

        // shift block over
        wi.userRangeStart = userRangeStartPoints[nextColumn];
        wi.userRangeEnd = userRangeEndPoints[nextColumn];
      }
    }

    latestError = rootMeanSquaredError(g);
    if (verifyPerIter) galois::gPrint("RMS: ", latestError, "\n");
    numIterations++;
  }
}

// TODO remove variants below? No concept of iteration + uses explicit locks
using SpinLock = galois::substrate::PaddedLock<true>;

void SGDSliceMarch(Graph& g, const LearnFN* lf) {
  unsigned numWorkItems = galois::getActiveThreads();
  ThreadWorkItem workItems[numWorkItems];

  unsigned int moviesPerThread = NUM_MOVIE_NODES / numWorkItems;
  unsigned int usersPerThread = NUM_USER_NODES / numWorkItems;

  unsigned int numSlices = NUM_USER_NODES / usersPerBlockSlice;
  unsigned int slicesPerThread = numSlices / numWorkItems;

  galois::gPrint("numSlices: ", numSlices,
                 "slicesPerThread: ", slicesPerThread, "\n");

  // set up initial work ranges for each thread
  for (unsigned int i = 0; i < numWorkItems; i++) {
    ThreadWorkItem wi;
    wi.movieRangeStart = moviesPerThread * i;
    wi.userRangeStart = usersPerThread * i;
    wi.userRangeEnd = NUM_USER_NODES;

    if (i == numWorkItems - 1) { // last block takes the rest
      wi.movieRangeEnd = NUM_MOVIE_NODES;
    } else {
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
  galois::do_all(galois::iterate(workItems + 0, workItems + numWorkItems), AdvanceEdgeOffsets(g));

  SpinLock* locks = new SpinLock[numSlices];
  galois::do_all(galois::iterate(workItems + 0, workItems + numWorkItems), 
    [&g, locks, step_size] (ThreadWorkItem& workItem) {
      galois::Timer timer;
      timer.start();
      unsigned int updates = 0;
      unsigned int conflicts = 0;

      unsigned int usersPerBlockSlice = workItem.usersPerBlockSlice;
      unsigned int currentBlockSliceEnd = workItem.userRangeStart;
      unsigned int userRangeEnd = workItem.userRangeEnd;

      unsigned int currentSliceId = workItem.sliceStart;
      unsigned int sliceUpdates = 0;

      while(sliceUpdates < maxUpdates * workItem.numSlices) {
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
          Graph::edge_iterator edge_it = g.edge_begin(movie, galois::MethodFlag::UNPROTECTED) + movie_data.edge_offset;
          Graph::edge_iterator edge_end = g.edge_end(movie, galois::MethodFlag::UNPROTECTED);
          for(;edge_it != edge_end; ++edge_it, ++movie_data.edge_offset)
          {
            GNode user = g.getEdgeDst(edge_it);

            //stop when you're outside the current block's user range
            if(user > currentBlockSliceEndUserId)
              break;

            Node& user_data = g.getData(user, galois::MethodFlag::UNPROTECTED);
            int edge_rating = g.getEdgeData(edge_it, galois::MethodFlag::UNPROTECTED);  

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
  );

  /*for(unsigned int i = 0; i < numWorkItems; i++)
    {
    ThreadWorkItem& workItem = workItems[i];
    printf("Worker %d took %lu to do %lu updates.\n", workItem.id, workItem.timeTaken/1000000, workItem.updates);
    }*/
}

typedef struct SliceInfo {
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

  void print() {
    printf("id: %d x: %d y: %d userStart: %d userEnd: %d movieStart: %d movieEnd: %d updates: %d\n", 
        id, x, y, userStart, userEnd, movieStart, movieEnd, updates);
  }
} SliceInfo;


struct calculate_user_offsets {
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
    Graph::edge_iterator edge_it = g.edge_begin(movie, galois::MethodFlag::UNPROTECTED), 
               edge_end = g.edge_end(movie, galois::MethodFlag::UNPROTECTED);

    for(unsigned int i = 0, offset = 0; i < numXSlices; i++, s++)
    {
      GNode user = g.getEdgeDst(edge_it);

      // in the slice's range
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

struct sgd_slice_jump {
  Graph& g;
  galois::GAccumulator<size_t> *eVisited;
  SpinLock *xLocks, *yLocks;
  SliceInfo* slices;
  unsigned numXSlices, numYSlices, numSlices;
  double step_size;
  sgd_slice_jump(Graph& g, galois::GAccumulator<size_t>* _eVisited, SpinLock* _xLocks, SpinLock* _yLocks, SliceInfo* _slices, 
      unsigned _numXSlices, unsigned _numYSlices, double _step_size) : 
    g(g), eVisited(_eVisited), xLocks(_xLocks), yLocks(_yLocks), slices(_slices), 
    numXSlices(_numXSlices), numYSlices(_numYSlices), numSlices(_numXSlices * _numYSlices),
    step_size(_step_size) {}
  
  //Preconditions: row and column of slice are locked
  //Postconditions: increments update count, does sgd update on each movie and user in the slice
  inline unsigned runSlice(SliceInfo* sp) const {
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
    for (; movie_it != end_movie_it; ++movie_it, ++movie_num) { 
      if(si.userOffsets[movie_num] < 0)
        continue;

      //get movie data
      GNode movie = *movie_it;
      Node& movie_data = g.getData(movie);

      unsigned int currentBlockSliceEndUserId = si.userEnd + NUM_MOVIE_NODES;

      //for each edge in the range
      Graph::edge_iterator edge_it = g.edge_begin(movie, galois::MethodFlag::UNPROTECTED) + si.userOffsets[movie_num];
      Graph::edge_iterator edge_end = g.edge_end(movie, galois::MethodFlag::UNPROTECTED);
      for(;edge_it != edge_end; ++edge_it, ++movie_data.edge_offset)
      {
        GNode user = g.getEdgeDst(edge_it);

        //stop when you're outside the current block's user range
        if(user > currentBlockSliceEndUserId)
          break;

        Node& user_data = g.getData(user, galois::MethodFlag::UNPROTECTED);
        int edge_rating = g.getEdgeData(edge_it, galois::MethodFlag::UNPROTECTED);  

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
  inline int getNextSlice(SliceInfo* sp) const {
    bool foundWork = false;
    unsigned workSearchTries = 0;
    unsigned nextSliceId = sp->id;
    while(foundWork || workSearchTries < 2 * numSlices)
    {
      workSearchTries++;

      nextSliceId++;
      if(nextSliceId == numSlices) nextSliceId = 0; //wrap around

      SliceInfo& nextSlice = slices[nextSliceId];

      if(nextSlice.updates < maxUpdates && xLocks[nextSlice.x].try_lock())
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

  void operator()(SliceInfo* startSlice) const {
    galois::Timer timer;
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
    *eVisited += tot_updates;
  }
};


void runSliceJump(Graph& g, const LearnFN* lf) {
  //set up parameters
  const unsigned threadCount = galois::getActiveThreads();
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
  galois::do_all(galois::iterate(g.begin(), g.begin() + NUM_MOVIE_NODES),
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
  galois::GAccumulator<size_t> eVisited;
  galois::do_all(galois::iterate(&startSlices[0], &startSlices[threadCount]),
      sgd_slice_jump(g, &eVisited, xLocks, yLocks, slices, numXSlices, numYSlices, step_size));

  galois::runtime::reportStat_Single("Matrix Completion", "EdgesVisited", eVisited.reduce());
}


template <typename A, typename G>
void runAlgo(const A& algo, G& g) {

  switch (learn) {
    case Intel:
      algo(g, IntelLearnFN());
      break;
    case Purdue:
      algo(g, PurdueLearnFN());
      break;
    case Bottou:
      algo(g, BottouLearnFN());
      break;
    case Inv:
      algo(g, InvLearnFN());
      break;
    default:
      std::abort();
      break;
  }
}


int main(int argc, char** argv) { 
  galois::SharedMemSys G;
  LonestarStart(argc, argv, name, desc, url);

  Graph g;

  // read structure of graph & edge weights
  galois::graphs::readGraph(g, inputFile);
  // fill each node's id & initialize the latent vectors
  std::tie(NUM_MOVIE_NODES, NUM_USER_NODES) = initializeGraphData(g);

  galois::gPrint("SUMMARY:",
                 "\nMovies: ", NUM_MOVIE_NODES,
                 "\nUsers: ", NUM_USER_NODES,
                 "\nRatings: ", g.sizeEdges(), 
                 "\nusersPerBlockSlice: ", usersPerBlockSlice, 
                 "\nmoviesPerBlockSlice: ", moviesPerBlockSlice, "\n");

  // shift edges if shiftFactor specified
  if (shiftFactor != 0) {
    for (auto ii = g.begin(), ee = g.end(); ii != ee; ++ii) {
      for (auto eii : g.edges(*ii)) g.getEdgeData(eii) -= shiftFactor;
    }
  }
  
  galois::StatTimer mainTimer;
  mainTimer.start();

  std::unique_ptr<LearnFN> lf;


  switch (algo) {
    case Algo::nodeMovie:
      runAlgo(&SGDNodeMovie, g);
      // SGDNodeMovie(g, lf.get());
      break;
    case Algo::edgeMovie:
      SGDEdgeMovie(g, lf.get());
      break;
    case Algo::block:
      runBlockSlices<SGDBlock>(g, lf.get());
      break;
    case Algo::blockAndSliceUsers:
      runBlockSlices<SGDBlockUsers>(g, lf.get());
      break;
    case Algo::blockAndSliceBoth:
      runBlockSlices<SGDBlockUsersMovies>(g, lf.get());
      break;
    case Algo::sliceMarch:
      SGDSliceMarch(g, lf.get());
      break;
    case Algo::sliceJump:
      runSliceJump(g, lf.get());
      break;
  }

  mainTimer.stop();

  return 0;
}
