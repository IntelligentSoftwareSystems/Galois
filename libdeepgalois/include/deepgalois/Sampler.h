#pragma once
#include "deepgalois/GraphTypes.h"

namespace deepgalois {
#define ETA 1.5          // length factor of DB in sampling
#define SAMPLE_CLIP 3000 // clip degree in sampling
#define DEFAULT_SIZE_FRONTIER 1000
#define DEFAULT_SIZE_SUBG 9000

class Sampler {
public:
  typedef int db_t;

protected:
  int m_;
  size_t count_;

  //! averaged degree of masked graph
  int avg_deg;
  //! average degree cut off to a clip
  int subg_deg;

  //VertexList vertices_;
  //mask_t* masks_;

  //! List of training nodes; sampling set
  std::vector<index_t> trainingNodes;

  //! masked original graph; typically to the training set
  Graph* globalMaskedGraph;
  Graph* globalGraph;
  DGraph* partGraph;

  //! Reindex a graph to only contain those in the vertex set
  void reindexSubgraph(VertexSet& keptVertices, Graph& g, Graph& reindexed);

  //! Given a graph, return a graph with edges to unmasked vertices removed in
  //! mg
  template <typename GraphTy>
  void getMaskedGraph(size_t n, mask_t* masks, GraphTy* g, Graph& sub) {
    std::vector<uint32_t> degrees(n, 0);
    this->getMaskedDegrees(n, masks, g, degrees);
    // auto offsets = deepgalois::parallel_prefix_sum(degrees);
    auto offsets = deepgalois::prefix_sum(degrees);
    size_t ne    = offsets[n];
    //galois::gPrint("Generate masked graph: num_vertices=", n, ", num_edges=", ne, "\n");
  
    // note this constructs the full graph's nodes; just trims edges
    sub.allocateFrom(n, ne);
    sub.constructNodes();
  
    galois::do_all(
        galois::iterate((size_t)0, n),
        [&](const auto src) {
          sub.fixEndEdge(src, offsets[src + 1]);
          if (masks[src] == 1) {
            auto idx = offsets[src];
            for (auto e = g->edge_begin(src); e != g->edge_end(src); e++) {
              const auto dst = g->getEdgeDst(e);
              if (masks[dst] == 1) {
                //galois::gPrint(src, " ", dst, "\n");
                sub.constructEdge(idx++, dst, 0);
              }
            }
          }
        }
        ,
        galois::loopname("gen_subgraph"));
  }


//! determine degree of each vertex in a masked graph (given by masks and g)
template <typename GraphTy>
void getMaskedDegrees(size_t n, mask_t* masks, GraphTy* g,
                                 std::vector<uint32_t>& degrees) {
  assert(degrees.size() == n);
#ifdef PARALLEL_GEN
  galois::do_all(
      galois::iterate(size_t(0), n),
      [&](const auto src) {
#else
  for (size_t src = 0; src < n; src++) {
#endif
        if (masks[src] == 1) {
          for (auto e = g->edge_begin(src); e != g->edge_end(src); e++) {
            const auto dst = g->getEdgeDst(e);
            if (masks[dst] == 1) {
              //galois::gInfo("Edge ", src, " ", dst);
              degrees[src]++;
            }
          }
        }
      }
#ifdef PARALLEL_GEN
      ,
      galois::loopname("update_degrees"));
#endif
}

  //! Set masks bitset with IDs in the vertices VertexSet
  void createMasks(size_t n, VertexSet vertices, mask_t* masks);
  inline VertexList reindexVertices(size_t n, VertexSet vertex_set);
  void checkGSDB(std::vector<db_t>& DB0, std::vector<db_t>& DB1,
                 std::vector<db_t>& DB2, size_t size);

  //! convert set of gids to lids
  VertexSet convertToLID(VertexSet& gidSet);

public:
  Sampler() : m_(DEFAULT_SIZE_FRONTIER) {}
  ~Sampler() {}

  //! sample a subgraph sg of size n from graph g
  //! sg is overwritten/is output
  void sampleSubgraph(size_t n, Graph& sg, mask_t* masks, unsigned seed = 0);

  //! API function for user-defined selection strategy
  // TODO how to expose this?
  virtual void selectVertices(size_t nv, size_t n, int m, Graph* g,
                               VertexList vertices, VertexSet& vertex_set);
  virtual void selectVertices(size_t n, int m, VertexSet& vertex_set,
                               unsigned seed);

  // galois::runtime::iterable<galois::NoDerefIterator<edge_iterator> >
  // neighbor_sampler(Graph &g, VertexID v);

  //! Given a mask, construct the graph with only those vertices ans ave as the
  //! masked graph in this class for the sampler.
  void initializeMaskedGraph(size_t count, mask_t* masks, Graph* g, DGraph* dg);
};

} // namespace deepgalois
