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
  index_t m; // number of vertice in the frontier
  size_t count_;

  //! averaged degree of masked graph
  int avg_deg;
  //! average degree cut off to a clip
  int subg_deg;

  // VertexList vertices_;
  // mask_t* masks_;

  //! List of training nodes; sampling set
  std::vector<index_t> trainingNodes;

  //! masked original graph; typically to the training set
  GraphCPU* globalMaskedGraph;
  GraphCPU* globalGraph;
  DGraph* partGraph;

  //! Reindex a graph to only contain those in the vertex set
  void reindexSubgraph(VertexSet& keptVertices, Graph& g, Graph& reindexed);

  //! Given a graph, return a graph with edges to unmasked vertices removed in mg
  template <typename GraphTy, typename SubgraphTy = Graph>
  void getMaskedGraph(index_t n, mask_t* masks, GraphTy* g, SubgraphTy* sub);

  //! determine degree of each vertex in a masked graph (given by masks and g)
  template <typename GraphTy = GraphCPU>
  void getMaskedDegrees(size_t n, mask_t* masks, GraphTy* g, std::vector<uint32_t>& degrees);

  //! Set masks bitset with IDs in the vertices VertexSet
  void createMasks(size_t n, VertexSet vertices, mask_t* masks);
  inline VertexList reindexVertices(size_t n, VertexSet vertex_set);
  //void checkGSDB(std::vector<db_t>& DB0, std::vector<db_t>& DB1, std::vector<db_t>& DB2, index_t size);

  //! convert set of gids to lids
  VertexSet convertToLID(VertexSet& gidSet);

  //! helper function to get degree of some vertex given some graph
  inline unsigned getDegree(GraphCPU* g, index_t v) {
    return g->edge_end(v) - g->edge_begin(v);
  }

  // helper function for graph saint implementation below
  void checkGSDB(std::vector<db_t>& DB0, std::vector<db_t>& DB1,
      std::vector<db_t>& DB2, index_t size) {
    if (DB0.capacity() < size) {
      DB0.reserve(DB0.capacity() * 2);
      DB1.reserve(DB1.capacity() * 2);
      DB2.reserve(DB2.capacity() * 2);
    }
    DB0.resize(size);
    DB1.resize(size);
    DB2.resize(size);
  }

public:
  Sampler() : m(DEFAULT_SIZE_FRONTIER) {}
  ~Sampler() {}

  //! sample a subgraph sg of size n from graph g
  //! sg is overwritten/is output
  void generateSubgraph(VertexSet &vertex_set, mask_t* masks, Graph* sg);

  //! API function for user-defined selection strategy
  // TODO how to expose this?
  virtual void selectVertices(index_t nv, index_t n, Graph* g, VertexList vertices, VertexSet& vertex_set);
  virtual void selectVertices(index_t n, VertexSet& vertex_set, unsigned seed);

  // galois::runtime::iterable<galois::NoDerefIterator<edge_iterator> >
  // neighbor_sampler(Graph &g, VertexID v);

  //! Given a mask, construct the graph with only those vertices ans ave as the
  //! masked graph in this class for the sampler.
  void initializeMaskedGraph(size_t count, mask_t* masks, GraphCPU* g, DGraph* dg);
};

} // namespace deepgalois
