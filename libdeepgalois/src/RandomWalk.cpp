#include <time.h>
#include <vector>
#include <iostream>
#include "galois/Galois.h"
#include "deepgalois/utils.h"
#include "deepgalois/Sampler.h"

namespace deepgalois {

void Sampler::initializeMaskedGraph(size_t count, mask_t* masks, GraphCPU* g, DGraph* dg) {
  this->count_ = count;
  // save original graph
  Sampler::globalGraph = g;
  // save partitioned graph
  Sampler::partGraph = dg;

  // allocate the object for the new masked graph
  Sampler::globalMaskedGraph = new GraphCPU();

  std::vector<uint32_t> degrees(g->size(), 0);
  // get degrees of nodes that will be in new graph
  //this->getMaskedDegrees(g->size(), masks, g, degrees);
  galois::do_all(galois::iterate(size_t(0), g->size()), [&](const auto src) {
    if (masks[src] == 1) {
      for (auto e = g->edge_begin(src); e != g->edge_end(src); e++) {
        const auto dst = g->getEdgeDst(e);
        if (masks[dst] == 1) degrees[src]++;
      }
    }
  } , galois::loopname("update_degrees"));

  auto offsets = deepgalois::parallel_prefix_sum(degrees);
  auto ne    = offsets[g->size()];

  // save ids (of original graph) of training nodes to vector
  for (size_t i = 0; i < g->size(); i++) {
    if (masks[i] == 1)
      Sampler::trainingNodes.push_back(i);
  }

  Sampler::globalMaskedGraph->allocateFrom(g->size(), ne);
  Sampler::globalMaskedGraph->constructNodes();
  // same as original graph, except keep only edges involved in masks
  galois::do_all(galois::iterate((size_t)0, g->size()), [&](const auto src) {
    Sampler::globalMaskedGraph->fixEndEdge(src, offsets[src + 1]);
    if (masks[src] == 1) {
      auto idx = offsets[src];
      for (auto e = g->edge_begin(src); e != g->edge_end(src); e++) {
        const auto dst = g->getEdgeDst(e);
        if (masks[dst] == 1) {
          // galois::gPrint(src, " ", dst, "\n");
          Sampler::globalMaskedGraph->constructEdge(idx++, dst, 0);
        }
      }
    }
  }, galois::loopname("gen_subgraph"));

  Sampler::globalMaskedGraph->degree_counting();
  Sampler::avg_deg = globalMaskedGraph->sizeEdges() / globalMaskedGraph->size();
  Sampler::subg_deg = (avg_deg > SAMPLE_CLIP) ? SAMPLE_CLIP : avg_deg;

  // TODO masked part graph as well to save time later; right now constructing
  // from full part graph
}

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

} // namespace deepgalois
