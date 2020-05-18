#include "galois/Galois.h"
#include "galois/gstl.h"
#include "galois/Reduction.h"
#include "galois/Timer.h"
#include "galois/Timer.h"
#include "galois/graphs/LCGraph.h"
#include "galois/graphs/TypeTraits.h"
#include "llvm/Support/CommandLine.h"
#include "galois/AtomicHelpers.h"

#include <iostream>
#include <fstream>
#include <deque>
#include <type_traits>

#include <random>
#include <math.h>
#include <algorithm>

#include "Lonestar/BoilerPlate.h"
#include "edge2vec.h"
#include "galois/DynamicBitset.h"

namespace cll = llvm::cl;

static const char* name = "edge2vec";

static const char* desc = "Preprocessing part of Node2vec";
static const char* url  = "edge2vec";
static cll::opt<std::string> filename(cll::Positional,
                                      cll::desc("<input file>"), cll::Required);
static cll::opt<uint32_t> N("N", cll::desc("Number of iterations"),
                            cll::init(10));

static cll::opt<uint32_t> walk_length("Walk Length",
                                      cll::desc("Length of random walk"),
                                      cll::init(50));

static cll::opt<double> p("p from the paper", cll::desc("p from the paper"),
                          cll::init(1));
static cll::opt<double> q("q from the paper", cll::desc("q"), cll::init(1));
static cll::opt<double> num_walk("number of walk", cll::desc("number of walk"),
                                 cll::init(1));

void computeVectors(std::vector<std::vector<uint32_t>>& v,
                    galois::InsertBag<std::vector<uint32_t>>& walks,
                    uint32_t num_edge_types) {

  galois::InsertBag<std::vector<uint32_t>> bag;
  galois::do_all(galois::iterate(walks), [&](std::vector<uint32_t>& walk) {
    std::vector<uint32_t> vec(num_edge_types + 1, 0);

    for (auto type : walk)
      vec[type]++;

    bag.push(vec);
  });

  for (auto vec : bag)
    v.push_back(vec);
}

void transformVectors(std::vector<std::vector<uint32_t>>& v,
                      std::vector<std::vector<uint32_t>>& transformedV,
                      uint32_t num_edge_types) {

  uint32_t rows = v.size();

  for (uint32_t j = 0; j <= num_edge_types; j++)
    for (uint32_t i = 0; i < rows; i++) {
      transformedV[j].push_back(v[i][j]);
    }
}

double sigmoidCal(double pears) {
  return 1 / (1 + exp(-pears)); // exact sig
  // return (pears / (1 + abs(pears))); //fast sigmoid
}

void computeMeans(std::vector<std::vector<uint32_t>>& v,
                  std::vector<double>& means, uint32_t num_edge_types) {

  galois::do_all(galois::iterate((uint32_t)1, num_edge_types + 1),
                 [&](uint32_t i) {
                   uint32_t sum = 0;
                   for (auto n : v[i])
                     sum += n;

                   means[i] = ((double)sum) / (v[i].size());
                 });
}
double pearsonCorr(uint32_t i, uint32_t j,
                   std::vector<std::vector<uint32_t>>& v,
                   std::vector<double>& means) {
  // int sum_x = 0, sum_y = 0, sum_xy = 0, squareSum_x = 0, squareSum_y = 0;
  std::vector<uint32_t> x = v[i];
  std::vector<uint32_t> y = v[j];

  double sum  = 0.0f;
  double sig1 = 0.0f;
  double sig2 = 0.0f;

  for (uint32_t m = 0; m < x.size(); m++) {
    sum += ((double)x[m] - means[i]) * ((double)y[m] - means[j]);
    // sum_x = sum_x + x.at(m);
    // sum_y = sum_y + y.at(m);
    // sum_xy = sum_xy + x.at(m) * y.at(m);
    // squareSum_x = squareSum_x + x.at(m) * x.at(m);
    // squareSum_x = squareSum_y + y.at(m) * y.at(m);
    sig1 += ((double)x[m] - means[i]) * ((double)x[m] - means[i]);
    sig2 += ((double)y[m] - means[j]) * ((double)y[m] - means[j]);
  }

  sum = sum / ((double)x.size());

  sig1 = sig1 / ((double)x.size());
  sig1 = sqrt(sig1);

  sig2 = sig2 / ((double)x.size());
  sig2 = sqrt(sig2);

  // double corr = (double)(x.size() * sum_xy - sum_x * sum_y)
  //            / sqrt((x.size() * squareSum_x - sum_x * sum_x)
  //              * (x.size() * squareSum_y - sum_y * sum_y));
  //
  double corr = sum / (sig1 * sig2);
  return corr;
}

void computeM(std::vector<std::vector<uint32_t>>& v, std::vector<double>& means,
              std::vector<std::vector<double>>& M, uint32_t num_edge_types) {

  galois::do_all(galois::iterate((uint32_t)1, num_edge_types + 1),
                 [&](uint32_t i) {
                   for (uint32_t j = 1; j <= num_edge_types; j++) {

                     double pearson_corr = pearsonCorr(i, j, v, means);
                     double sigmoid      = sigmoidCal(pearson_corr);

                     M[i][j] = sigmoid;
                   }
                 });
}
// function HeteroRandomWalk
void heteroRandomWalk(Graph& graph, std::vector<std::vector<double>>& M,
                      galois::InsertBag<std::vector<uint32_t>>& bag,
                      galois::InsertBag<std::vector<uint32_t>>& nodeWalks,
                      uint32_t walk_length, double p, double q) {

  galois::do_all(
      galois::iterate(graph),
      [&](GNode n) {
        std::vector<uint32_t> walk;
        std::vector<uint32_t> T;

        walk.push_back(n);
        while (walk.size() < walk_length) {

          if (walk.size() == 1) {

            // uint32_t curr = walk[0];

            // random value between 0 and 1
            double prob = distribution(generator);

            // sample a neighbor of curr
            uint32_t m, type;
            uint32_t total_wt = 0;
            for (auto e : graph.edges(n)) {

              //	uint32_t v = graph.getEdgeDst(e);
              uint32_t wt = graph.getEdgeData(e).weight;

              total_wt += wt;
            }

            prob     = prob * total_wt;
            total_wt = 0;
            for (auto e : graph.edges(n)) {

              uint32_t wt = graph.getEdgeData(e).weight;
              total_wt += wt;

              if (total_wt >= prob) {
                m    = graph.getEdgeDst(e);
                type = graph.getEdgeData(e).type;
                break;
              }
            }

            walk.push_back(m);
            T.push_back(type);

          } else {
            uint32_t curr = walk[walk.size() - 1];
            uint32_t prev = walk[walk.size() - 2];

            uint32_t p1 = T.back(); // last element of T

            double total_ew = 0.0f;

            std::vector<double> vec_ew;
            std::vector<uint32_t> neighbors;
            std::vector<uint32_t> types;

            for (auto k : graph.edges(curr)) {

              uint32_t p2 = graph.getEdgeData(k).type;

              double alpha;
              uint32_t next = graph.getEdgeDst(k);

              if (next == prev)
                alpha = 1.0f / p;
              else if (graph.findEdge(next, prev) != graph.edge_end(next) ||
                       graph.findEdge(prev, next) != graph.edge_end(prev))
                alpha = 1.0f;
              else
                alpha = 1.0f / q;

              double ew =
                  M[p1][p2] * ((double)graph.getEdgeData(k).weight) * alpha;

              total_ew += ew;
              neighbors.push_back(next);
              vec_ew.push_back(ew);
              types.push_back(p2);
            }

            // randomly sample a neighobr of curr
            // random value between 0 and 1
            double prob = distribution(generator);
            prob        = prob * total_ew;
            total_ew    = 0.0f;

            // sample a neighbor of curr
            uint32_t m, type;

            int32_t idx = 0;
            for (auto k : neighbors) {
              total_ew += vec_ew[idx];
              if (total_ew >= prob) {
                m    = k;
                type = types[idx];
                break;
              }
              idx++;
            }

            walk.push_back(m);
            T.push_back(type);
          } // end if-else loop
        }   // end while

        nodeWalks.push(walk);
        bag.push(T);
      },
      galois::steal());
}

void printM(std::vector<std::vector<double>>& M, uint32_t num_edge_types) {

  for (uint32_t i = 0; i <= num_edge_types; i++) {
    for (uint32_t j = 0; j <= num_edge_types; j++)
      std::cout << M[i][j] << " ";

    std::cout << std::endl;
  }
}

// function generateTransitionMatrix
// M should have all entries set to 1
void generateTransitionMatrix(
    Graph& graph, std::vector<std::vector<double>>& M, uint32_t N,
    uint32_t walk_length, double p, double q, uint32_t num_edge_types,
    galois::InsertBag<std::vector<uint32_t>>& nodeWalks) {

  std::cout << "legth:" << walk_length << std::endl;
  while (N > 0) {
    std::cout << "N: " << N << std::endl;
    N--;

    galois::StatTimer T("walk");
    T.start();
    // E step; generate walks
    nodeWalks.clear();
    galois::InsertBag<std::vector<uint32_t>> walks;
    heteroRandomWalk(graph, M, walks, nodeWalks, walk_length, p, q);
    T.stop();
    std::cout << "walk time: " << T.get() << std::endl;

    // M step
    std::vector<std::vector<uint32_t>> v;
    computeVectors(v, walks, num_edge_types);

    std::vector<std::vector<uint32_t>> transformedV(num_edge_types + 1);
    transformVectors(v, transformedV, num_edge_types);

    std::vector<double> means(num_edge_types + 1);
    computeMeans(transformedV, means, num_edge_types);

    computeM(transformedV, means, M, num_edge_types);

    printM(M, num_edge_types);
  }
}

void printWalks(galois::InsertBag<std::vector<uint32_t>>& walks) {

  std::ofstream f("walks.txt");

  for (auto walk : walks) {
    for (auto node : walk)
      f << node + 1 << " ";
    f << std::endl;
  }
}

int main(int argc, char** argv) {
  galois::SharedMemSys G;
  LonestarStart(argc, argv, name, desc, url);

  Graph graph;

  std::ifstream f(filename.c_str());
  // read graph

  uint32_t nodes;
  uint64_t edges;

  std::string line;
  std::getline(f, line);
  std::stringstream ss(line);

  ss >> edges >> nodes;

  std::vector<std::vector<uint32_t>> edges_id(nodes);
  std::vector<std::vector<EdgeTy>> edges_data(nodes);
  std::vector<uint64_t> prefix_edges(nodes);

  uint64_t max_type = 0;

  while (std::getline(f, line)) {

    std::stringstream ss(line);
    uint32_t src, dst, type, id;
    ss >> src >> dst >> type >> id;

    edges_id[src - 1].push_back(dst - 1);
    edges_id[dst - 1].push_back(src - 1);

    EdgeTy edgeTy1, edgeTy2;

    edgeTy1.weight = 1;
    edgeTy1.type   = type;

    edgeTy2.weight = 1;
    edgeTy2.type   = type;

    if (type > max_type)
      max_type = type;

    edges_data[src - 1].push_back(edgeTy1);
    edges_data[dst - 1].push_back(edgeTy2);
  }

  f.close();

  galois::do_all(galois::iterate(uint32_t{0}, nodes),
                 [&](uint32_t c) { prefix_edges[c] = edges_id[c].size(); });

  for (uint32_t c = 1; c < nodes; ++c) {
    prefix_edges[c] += prefix_edges[c - 1];
  }

  graph.constructFrom(nodes, 2 * edges, prefix_edges, edges_id, edges_data);

  galois::StatTimer T("end2end");
  T.start();

  // transition matrix
  std::vector<std::vector<double>> M(max_type + 1);

  // initialize transition matrix
  for (uint32_t i = 0; i <= max_type; i++) {
    for (uint32_t j = 0; j <= max_type; j++)
      M[i].push_back(1.0f);
  }

  galois::InsertBag<std::vector<uint32_t>> walks;
  generateTransitionMatrix(graph, M, N, walk_length, p, q, max_type, walks);

  T.stop();
  std::cout << "total time:" << T.get() << std::endl;

  printWalks(walks);

  return 0;
}
