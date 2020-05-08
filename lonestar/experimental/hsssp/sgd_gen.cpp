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
#include <vector>
#include <limits>
#include "galois/Galois.h"
#include "Lonestar/BoilerPlate.h"
#include "galois/gstl.h"

#include "galois/runtime/CompilerHelperFunctions.h"

#include "galois/graphs/OfflineGraph.h"
#include "DistGraph.h"

static const char* const name =
    "SGD - Compiler Generated Distributed Heterogeneous";
static const char* const desc =
    "Stochastic gradient descent on Distributed Galois.";
static const char* const url = 0;

namespace cll = llvm::cl;
static cll::opt<std::string>
    inputFile(cll::Positional, cll::desc("<input file>"), cll::Required);
static cll::opt<unsigned int> maxIterations("maxIterations",
                                            cll::desc("Maximum iterations"),
                                            cll::init(4));
static cll::opt<unsigned int>
    src_node("startNode", cll::desc("ID of the source node"), cll::init(0));
static cll::opt<bool>
    verify("verify",
           cll::desc("Verify ranks by printing to 'page_ranks.#hid.csv' file"),
           cll::init(false));

#define LATENT_VECTOR_SIZE 2
static const double LEARNING_RATE = 0.001; // GAMMA, Purdue: 0.01 Intel: 0.001
static const double DECAY_RATE    = 0.9;   // STEP_DEC, Purdue: 0.1 Intel: 0.9
static const double LAMBDA        = 0.001; // Purdue: 1.0 Intel: 0.001

struct SGD_NodeData {
  // double latent_vector[LATENT_VECTOR_SIZE];
  SGD_NodeData() : latent_vector(LATENT_VECTOR_SIZE) {}
  std::vector<double> latent_vector;
  uint64_t updates;
  uint64_t edge_offset;
};

typedef DistGraph<SGD_NodeData, int> Graph;
typedef typename Graph::GraphNode GNode;

static double genRand() {
  // generate a random double in (-1,1)
  return 2.0 * ((double)std::rand() / (double)RAND_MAX) - 1.0;
}

double getstep_size(unsigned int round) {
  return LEARNING_RATE * 1.5 / (1.0 + DECAY_RATE * pow(round + 1, 1.5));
}

struct InitializeGraph {
  Graph* graph;

  void static go(Graph& _graph) {

    struct SyncerPull_0 {
      static unsigned long extract(const struct SGD_NodeData& node) {
        return node.updates;
      }
      static void setVal(struct SGD_NodeData& node, unsigned long y) {
        node.updates = y;
      }
      typedef unsigned long ValTy;
    };
    struct SyncerPull_1 {
      static unsigned long extract(const struct SGD_NodeData& node) {
        return node.edge_offset;
      }
      static void setVal(struct SGD_NodeData& node, unsigned long y) {
        node.edge_offset = y;
      }
      typedef unsigned long ValTy;
    };
    struct SyncerPull_2 {
      static std::vector<double> extract(const struct SGD_NodeData& node) {
        return node.latent_vector;
      }
      static void setVal(struct SGD_NodeData& node, std::vector<double> y) {
        node.latent_vector = y;
      }
      typedef std::vector<double> ValTy;
    };
    galois::do_all(
        _graph.begin(), _graph.end(), InitializeGraph{&_graph},
        galois::loopname("SGD Init"),
        galois::write_set("sync_pull", "this->graph", "struct SGD_NodeData &",
                          "struct SGD_NodeData &", "updates", "unsigned long"),
        galois::write_set("sync_pull", "this->graph", "struct SGD_NodeData &",
                          "struct SGD_NodeData &", "edge_offset",
                          "unsigned long"),
        galois::write_set("sync_pull", "this->graph", "struct SGD_NodeData &",
                          "struct SGD_NodeData &", "latent_vector",
                          "std::vector<double>"));
    _graph.sync_pull<SyncerPull_0>();

    _graph.sync_pull<SyncerPull_1>();

    _graph.sync_pull<SyncerPull_2>();
  }

  void operator()(GNode src) const {
    SGD_NodeData& sdata = graph->getData(src);

    sdata.updates     = 0;
    sdata.edge_offset = 0;

    for (int i = 0; i < LATENT_VECTOR_SIZE; i++) {
      sdata.latent_vector[i] = genRand();
    }
  }
};

void dummy_func(SGD_NodeData& user, SGD_NodeData& movie) {
  for (int i = 0; i < LATENT_VECTOR_SIZE; ++i) {
    user.latent_vector[i]  = 3.0; // genRand();
    movie.latent_vector[i] = 3.0; // genRand();
  }
}

static const double MINVAL = -1e+100;
static const double MAXVAL = 1e+100;
double calcPrediction(const SGD_NodeData& movie_data,
                      const SGD_NodeData& user_data) {
  double pred = galois::innerProduct(movie_data.latent_vector.begin(),
                                     movie_data.latent_vector.begin(),
                                     user_data.latent_vector.begin(), 0.0);
  double p    = pred;
  pred        = std::min(MAXVAL, pred);
  pred        = std::max(MINVAL, pred);
  if (p != pred)
    std::cerr << "clamped " << p << " to " << pred << "\n";
  return pred;
}

struct Sgd {
  Graph* graph;
  double step_size;

  void static go(Graph& _graph, double _step_size) {

    struct Syncer_0 {
      static class std::vector<double, class std::allocator<double>>
      extract(const struct SGD_NodeData& node) {
        return node.latent_vector;
      } static void
      reduce(struct SGD_NodeData& node,
             class std::vector<double, class std::allocator<double>> y) {
        galois::pairWiseAvg_vec(node.latent_vector, y);
      }
      static void reset(struct SGD_NodeData& node) {
        galois::resetVec(node.latent_vector);
      }
      typedef class std::vector<double, class std::allocator<double>> ValTy;
    };
    struct SyncerPull_0 {
      static class std::vector<double, class std::allocator<double>>
      extract(const struct SGD_NodeData& node) {
        return node.latent_vector;
      } static void
      setVal(struct SGD_NodeData& node,
             class std::vector<double, class std::allocator<double>> y) {
        node.latent_vector = y;
      }
      typedef class std::vector<double, class std::allocator<double>> ValTy;
    };
    galois::do_all(
        _graph.begin(), _graph.end(), Sgd{&_graph, _step_size},
        galois::loopname("SGD"),
        galois::write_set(
            "sync_push", "this->graph", "struct SGD_NodeData &",
            "std::vector<class std::vector<double, class "
            "std::allocator<double> >>",
            "latent_vector",
            "class std::vector<double, class std::allocator<double> >",
            "{galois::pairWiseAvg_vec(node.latent_vector, y); }",
            "{galois::resetVec(node.latent_vector); }"),
        galois::write_set(
            "sync_pull", "this->graph", "struct SGD_NodeData &",
            "std::vector<class std::vector<double, class "
            "std::allocator<double> >>",
            "latent_vector",
            "class std::vector<double, class std::allocator<double> >"));
    _graph.sync_push<Syncer_0>();

    _graph.sync_pull<SyncerPull_0>();
  }

  void operator()(GNode src) const {
    SGD_NodeData& sdata = graph->getData(src);
    auto& movie_node    = sdata.latent_vector;

    for (auto jj = graph->edge_begin(src), ej = graph->edge_end(src); jj != ej;
         ++jj) {
      GNode dst          = graph->getEdgeDst(jj);
      auto& ddata        = graph->getData(dst);
      auto& user_node    = ddata.latent_vector;
      auto& sdata_up     = sdata.updates;
      double edge_rating = graph->getEdgeData(dst);

      // doGradientUpdate
      double old_dp = galois::innerProduct(user_node.begin(), user_node.end(),
                                           movie_node.begin(), 0.0);
      double cur_error = edge_rating - old_dp;
      assert(cur_error < 1000 && cur_error > -1000);
      for (int i = 0; i < LATENT_VECTOR_SIZE; ++i) {
        double prevUser  = user_node[i];
        double prevMovie = movie_node[i];

        user_node[i] += step_size * (cur_error * prevMovie - LAMBDA * prevUser);
        assert(std::isnormal(user_node[i]));
        movie_node[i] +=
            step_size * (cur_error * prevUser - LAMBDA * prevMovie);
        assert(std::isnormal(movie_node[i]));
      }
      double cur_error2 = edge_rating - calcPrediction(sdata, ddata);
      if (std::abs(cur_error - cur_error2) > 20) {
        // push.
        std::cerr << "A" << std::abs(cur_error - cur_error2) << "\n";
        // didWork = true;
      }
    }
  }
};

int main(int argc, char** argv) {
  try {

    LonestarStart(argc, argv, name, desc, url);
    auto& net = galois::runtime::getSystemNetworkInterface();

    OfflineGraph g(inputFile);

    Graph hg(inputFile, net.ID, net.Num);

    InitializeGraph::go(hg);

    if (verify) {
      for (auto i = hg.begin(); i != hg.end(); ++i) {
        std::cout << hg.getData(*i).latent_vector[0] << "\n";
      }
    }

    for (unsigned iter = 0; iter < maxIterations; ++iter) {
      auto step_size = getstep_size(iter);
      std::cout << " Iter :" << iter << " step_size :" << step_size << "\n";
      Sgd::go(hg, step_size);
    }

    if (verify) {
      if (net.ID == 0) {
        std::cout << "After sgd \n";
        for (auto i = hg.begin(); i != hg.end(); ++i) {
          std::cout << hg.getData(*i).latent_vector[0] << "\n";
        }
      }
    }

    return 0;
  } catch (const char* c) {
    std::cerr << "Error: " << c << "\n";
    return 1;
  }
}
