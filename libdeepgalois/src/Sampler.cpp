#include <time.h>
#include <vector>
#include "galois/Galois.h"
#include "deepgalois/utils.h"
#include "deepgalois/Sampler.h"
#define PARALLEL_GEN

namespace deepgalois {

//! debug function: prints out sets of vertices
void print_vertex_set(VertexSet vertex_set) {
  unsigned counter = 0;
  unsigned n       = vertex_set.size();
  galois::gPrint("( ");
  for (int i : vertex_set) {
    counter++;
    if (counter > 16 && counter < n - 16)
      continue;
    galois::gPrint(i, " ");
  }
  galois::gPrint(")\n");
}

/*
// implementation from GraphSAINT
// https://github.com/GraphSAINT/GraphSAINT/blob/master/ipdps19_cpp/sample.cpp
void Sampler::selectVertices(index_t n, VertexSet& st, unsigned seed) {
  if (n < m) m = n;
  unsigned myseed = seed;

  // unsigned myseed = tid;
  // DBx: Dashboard line x, IAx: Index array line x
  std::vector<db_t> DB0, DB1, DB2, IA0, IA1, IA2, IA3, IA4, nDB0, nDB1, nDB2;
  DB0.reserve(subg_deg * m * ETA);
  DB1.reserve(subg_deg * m * ETA);
  DB2.reserve(subg_deg * m * ETA);
  IA0.reserve(n);
  IA1.reserve(n);
  IA2.reserve(n);
  IA3.reserve(n);
  IA4.reserve(n);
  IA0.resize(m);
  IA1.resize(m);
  IA2.resize(m);
  IA3.resize(m);

  // galois::gPrint("seed ", myseed, " m ", m, "\n");
  // galois::gPrint("trainingNodes size: ", trainingNodes.size(), "\n");
  // printf("( ");
  // for (size_t i = 0; i < 10; i++) std::cout << trainingNodes[i] << " ";
  // printf(")\n");

  for (index_t i = 0; i < m; i++) {
    auto rand_idx = rand_r(&myseed) % Sampler::trainingNodes.size();
    db_t v = IA3[i] = Sampler::trainingNodes[rand_idx];
    st.insert(v);
    IA0[i] = getDegree(Sampler::globalMaskedGraph, v);
    IA0[i] = (IA0[i] > SAMPLE_CLIP) ? SAMPLE_CLIP : IA0[i];
    IA1[i] = 1;
    IA2[i] = 0;
  }
  // calculate prefix sum for IA0 and store in IA2 to compute the address for
  // each frontier in DB
  IA2[0] = IA0[0];
  for (index_t i = 1; i < m; i++)
    IA2[i] = IA2[i - 1] + IA0[i];
  // now fill DB accordingly
  checkGSDB(DB0, DB1, DB2, IA2[m - 1]);
  for (index_t i = 0; i < m; i++) {
    db_t DB_start = (i == 0) ? 0 : IA2[i - 1];
    db_t DB_end   = IA2[i];
    for (auto j = DB_start; j < DB_end; j++) {
      DB0[j] = IA3[i];
      DB1[j] = (j == DB_start) ? (j - DB_end) : (j - DB_start);
      DB2[j] = i + 1;
    }
  }

  db_t choose, neigh_v, newsize, tmp;
  for (index_t itr = 0; itr < n - m; itr++) {
    choose = db_t(-1);
    while (choose == db_t(-1)) {
      tmp = rand_r(&myseed) % DB0.size();
      if (size_t(tmp) < DB0.size())
        if (DB0[tmp] != db_t(-1))
          choose = tmp;
    }
    choose      = (DB1[choose] < 0) ? choose : (choose - DB1[choose]);
    db_t v      = DB0[choose];
    auto degree = getDegree(Sampler::globalMaskedGraph, v);
    neigh_v     = (degree != 0) ? rand_r(&myseed) % degree : db_t(-1);
    if (neigh_v != db_t(-1)) {
      neigh_v = Sampler::globalMaskedGraph->getEdgeDst(
          Sampler::globalMaskedGraph->edge_begin(v) + neigh_v);
      st.insert(neigh_v);
      IA1[DB2[choose] - 1] = 0;
      IA0[DB2[choose] - 1] = 0;
      for (auto i = choose; i < choose - DB1[choose]; i++)
        DB0[i] = db_t(-1);
      newsize = getDegree(Sampler::globalMaskedGraph, neigh_v);
      newsize = (newsize > SAMPLE_CLIP) ? SAMPLE_CLIP : newsize;
    } else
      newsize = 0;
    // shrink DB to remove sampled nodes, also shrink IA accordingly
    bool cond = DB0.size() + newsize > DB0.capacity();
    if (cond) {
      // compute prefix sum for the location in shrinked DB
      IA4.resize(IA0.size());
      IA4[0] = IA0[0];
      for (size_t i = 1; i < IA0.size(); i++)
        IA4[i] = IA4[i - 1] + IA0[i];
      nDB0.resize(IA4.back());
      nDB1.resize(IA4.back());
      nDB2.resize(IA4.back());
      IA2.assign(IA4.begin(), IA4.end());
      for (size_t i = 0; i < IA0.size(); i++) {
        if (IA1[i] == 0)
          continue;
        db_t DB_start = (i == 0) ? 0 : IA4[i - 1];
        db_t DB_end   = IA4[i];
        for (auto j = DB_start; j < DB_end; j++) {
          nDB0[j] = IA3[i];
          nDB1[j] = (j == DB_start) ? (j - DB_end) : (j - DB_start);
          nDB2[j] = i + 1;
        }
      }
      // remap the index in DB2 by compute prefix of IA1 (new idx in IA)
      IA4.resize(IA1.size());
      IA4[0] = IA1[0];
      for (size_t i = 1; i < IA1.size(); i++)
        IA4[i] = IA4[i - 1] + IA1[i];
      DB0.assign(nDB0.begin(), nDB0.end());
      DB1.assign(nDB1.begin(), nDB1.end());
      DB2.assign(nDB2.begin(), nDB2.end());
      for (auto i = DB2.begin(); i < DB2.end(); i++)
        *i = IA4[*i - 1];
      db_t curr = 0;
      for (size_t i = 0; i < IA0.size(); i++) {
        if (IA0[i] != 0) {
          IA0[curr] = IA0[i];
          IA1[curr] = IA1[i];
          IA2[curr] = IA2[i];
          IA3[curr] = IA3[i];
          curr++;
        }
      }
      IA0.resize(curr);
      IA1.resize(curr);
      IA2.resize(curr);
      IA3.resize(curr);
    }
    checkGSDB(DB0, DB1, DB2, newsize + DB0.size());
    IA0.push_back(newsize);
    IA1.push_back(1);
    IA2.push_back(IA2.back() + IA0.back());
    IA3.push_back(neigh_v);
    db_t DB_start = (*(IA2.end() - 2));
    db_t DB_end   = IA2.back();
    for (auto j = DB_start; j < DB_end; j++) {
      DB0[j] = IA3.back();
      DB1[j] = (j == DB_start) ? (j - DB_end) : (j - DB_start);
      DB2[j] = IA3.size();
    }
  }
  // galois::gPrint("Done selection, vertex_set size: ", st.size(), ", set: ");
  // print_vertex_set(st);
}
*/

// API function for user-defined selection strategy
// Select n vertices from vertices and put them in vertex_set.
// nv: number of vertices in the original graph;
// n: number of vertices in the subgraph;
// m: number of vertices in the frontier.
// our implementation of GraphSAINT sampling
void Sampler::selectVertices(index_t nv, index_t n, Graph* g,
                             VertexList vertices, VertexSet& vertex_set) {
  // galois::gPrint("Select a vertex set of size ", n, " from ", nv, " vertices,
  // graph size: ", g->size(), "\n");
  assert(nv == vertices.size());
  // randomly select m vertices from vertices as frontier
  auto frontier_indices = deepgalois::select_k_items((int)m, 0, (int)nv);
  VertexList frontier(m);
  for (index_t i = 0; i < m; i++)
    frontier[i] = vertices[frontier_indices[i]];
  vertex_set.insert(frontier.begin(), frontier.end());
  // galois::gPrint("vertex_set size: ", vertex_set.size(), "\n");
  int* degrees = new int[m];
  //galois::do_all(galois::iterate(size_t(0), size_t(m)), [&](const auto i) {
  for (index_t i = 0; i < m; i++) {
    degrees[i] = (int)getDegree(g, frontier[i]);
  } //, galois::loopname("compute_degrees"));
  for (index_t i = 0; i < n - m; i++) {
    auto pos    = select_one_item((int)m, degrees);
    auto u      = frontier[pos];
    auto degree = degrees[pos];
    int j       = 0;
    for (; j < degree; j++) {
      auto neighbor_id = rand() % degree; // randomly select a neighbor
      auto dst         = g->getEdgeDst(g->edge_begin(u) + neighbor_id);
      if (vertex_set.find(dst) == vertex_set.end()) {
        frontier[pos] = dst;
        degrees[pos]  = getDegree(g, frontier[pos]);
        vertex_set.insert(dst);
        break;
      }
    }
    if (j == degree)
      galois::gPrint("Not found from ", degree, " neighbors\n");
  }
  /*
  assert(n == vertex_set.size()); // size of vertex_set could be slightly
  smaller than n galois::gPrint("Done selection, vertex_set size: ",
  vertex_set.size(), ", set: "); print_vertex_set(vertex_set);
  */
}

void Sampler::createMasks(size_t n, VertexSet vertices, mask_t* masks) {
  // galois::gPrint("Updating masks, size = ", vertices.size(), "\n");
  std::fill(masks, masks + n, 0);
  for (auto v : vertices) masks[v] = 1;
}

inline VertexList Sampler::reindexVertices(size_t n, VertexSet vertex_set) {
  VertexList new_ids(n, 0);
  int vid = 0;
  for (auto v : vertex_set) {
    new_ids[v] = vid++; // reindex
  }
  return new_ids;
}

// Given a subset of vertices and a graph g, generate a subgraph sg from the
// graph g
void Sampler::reindexSubgraph(VertexSet& keptVertices, Graph& origGraph, Graph& reindexGraph) {
  // auto n = origGraph.size(); // old graph size
  auto nv            = keptVertices.size(); // new graph (subgraph) size
  VertexList new_ids = this->reindexVertices(globalGraph->size(), keptVertices);
  std::vector<uint32_t> degrees(nv, 0); // degrees of vertices in the subgraph
  for (auto v : keptVertices) {
    degrees[new_ids[v]] = getDegree(&origGraph, v);
  }
  // auto offsets = deepgalois::parallel_prefix_sum(degrees);
  auto offsets = deepgalois::prefix_sum(degrees);
  auto ne      = offsets[nv];
  // galois::gPrint("Generate subgraph: num_vertices=", nv, ", num_edges=", ne,
  // "\n");
  reindexGraph.allocateFrom(nv, ne);
  reindexGraph.constructNodes();
  VertexList old_ids(keptVertices.begin(),
                     keptVertices.end()); // vertex ID mapping
#ifdef PARALLEL_GEN
  galois::do_all(galois::iterate(size_t(0), size_t(nv)), [&](const auto i) {
#else
  for (size_t i = 0; i < nv; i++) {
#endif
        reindexGraph.fixEndEdge(i, offsets[i + 1]);
        unsigned j  = 0;
        auto old_id = old_ids[i];
        for (auto e = origGraph.edge_begin(old_id);
             e != origGraph.edge_end(old_id); e++) {
          auto dst = new_ids[origGraph.getEdgeDst(e)];
          assert(dst < nv);
          reindexGraph.constructEdge(offsets[i] + j, dst, 0);
          j++;
        }
      }
#ifdef PARALLEL_GEN
      , galois::loopname("construct_graph"));
#endif
}

VertexSet Sampler::convertToLID(VertexSet& gidSet) {
  VertexSet existingLIDs;
  // find local selected vertices, convert to lid
  for (auto i : gidSet) {
    if (partGraph->isLocal(i)) {
      existingLIDs.insert(partGraph->getLID(i));
    }
  }
  return existingLIDs;
}

template <typename GraphTy>
void Sampler::getMaskedDegrees(size_t n, mask_t* masks, GraphTy* g, std::vector<uint32_t>& degrees) {
//template <>
//void Sampler::getMaskedDegrees(size_t n, mask_t* masks, GraphCPU* g, std::vector<uint32_t>& degrees) {
  assert(degrees.size() == n);
  galois::do_all(galois::iterate(size_t(0), n), [&](const auto src) {
  //for (size_t src = 0; src < n; src++) {
    if (masks[src] == 1) {
      for (auto e = g->edge_begin(src); e != g->edge_end(src); e++) {
        const auto dst = g->getEdgeDst(e);
        if (masks[dst] == 1) {
          // galois::gInfo("Edge ", src, " ", dst);
          degrees[src]++;
        }
      }
    }
  } , galois::loopname("update_degrees"));
}

template <typename GraphTy, typename SubgraphTy>
void Sampler::getMaskedGraph(index_t n, mask_t* masks, GraphTy* g, SubgraphTy* sub) {
  std::vector<uint32_t> degrees(n, 0);
  this->getMaskedDegrees(n, masks, g, degrees);
  // auto offsets = deepgalois::parallel_prefix_sum(degrees);
  auto offsets = deepgalois::prefix_sum(degrees);
  size_t ne    = offsets[n];
  // galois::gPrint("getMaskedGraph: num_vertices=", n, ", num_edges=", ne, "\n");

  // note this constructs the full graph's nodes; just trims edges
  sub->allocateFrom(n, ne);
  sub->constructNodes();

  galois::do_all(galois::iterate(size_t(0), size_t(n)), [&](const auto src) {
    sub->fixEndEdge(src, offsets[src + 1]);
    if (masks[src] == 1) {
      auto idx = offsets[src];
      for (auto e = g->edge_begin(src); e != g->edge_end(src); e++) {
        auto dst = g->getEdgeDst(e);
        if (masks[dst] == 1) {
          // galois::gPrint(src, " ", dst, "\n");
          sub->constructEdge(idx++, dst, 0);
        }
      }
    }
  }, galois::loopname("gen_subgraph"));
}

void Sampler::generateSubgraph(VertexSet &sampledSet, mask_t* masks, Graph* sg) {
  // n = 9000 by default
  // do the sampling of vertices from training set + using masked graph

  // sampledSet is a list of *global* ids in the graph
  // create new vertex set with LIDs for partitioned graph
  VertexSet sampledLIDs = this->convertToLID(sampledSet);

  // VertexSet sampledLIDs;
  // galois::gPrint("part graph num edges is ", partGraph->sizeEdges(), "\n");
  // galois::gPrint("global mask num edges is ", globalMaskedGraph->sizeEdges(),
  // "\n"); for (auto i : this->trainingNodes) {
  //  sampledLIDs.insert(i);
  //}

  // create the masks
  createMasks(Sampler::partGraph->size(), sampledLIDs, masks);

  // this graph will contain sampled vertices and induced subgraph for it
  Graph maskedSG;
  // TODO use partMaskedGraph once constructed later
  // remove edges whose destination is not masked
  this->getMaskedGraph(Sampler::partGraph->size(), masks, Sampler::partGraph, &maskedSG);
  this->reindexSubgraph(sampledLIDs, maskedSG, *sg);

  // galois::gPrint("sg num edges is ", sg.sizeEdges(), "\n");
}

} // namespace deepgalois
