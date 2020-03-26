/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2020, The University of Texas at Austin. All rights reserved.
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

/**
 * CC w/ Afforest sampling.
 */
struct AfforestAlgo {
  struct NodeData : public galois::UnionFindNode<NodeData> {
    using component_type = NodeData*;

    NodeData() : galois::UnionFindNode<NodeData>(const_cast<NodeData*>(this)) {}
    NodeData(const NodeData& o) : galois::UnionFindNode<NodeData>(o.m_component) {}

    component_type component() { return this->get(); }
    bool isRepComp(unsigned int x) { return false; } // verify

  public:
    void link(NodeData* b) {
      NodeData* a = m_component.load(std::memory_order_relaxed);
      b = b->m_component.load(std::memory_order_relaxed);
      while (a != b) {
        if (a < b)
          std::swap(a, b);
        // Now a > b
        NodeData* ac = a->m_component.load(std::memory_order_relaxed);
        if (
          (ac == a && a->m_component.compare_exchange_strong(a, b))
          || (b == ac)
          )
          break;
        a = (a->m_component.load(std::memory_order_relaxed))->m_component.load(std::memory_order_relaxed);
        b = b->m_component.load(std::memory_order_relaxed);
      }
    }
  };
  using Graph =
      galois::graphs::LC_CSR_Graph<NodeData, void>::with_no_lockable<true>::type;
  using GNode = Graph::GraphNode;
  using component_type = NodeData::component_type;

  template <typename G>
  void readGraph(G& graph) {
    galois::graphs::readGraph(graph, inputFilename);
  }

  const uint32_t NUM_SAMPLES = 1024;
  component_type approxLargestComponent(Graph& graph) {

    using map_type = std::unordered_map<component_type, int,
        std::hash<component_type>, std::equal_to<component_type>,
        galois::gstl::Pow2Alloc<std::pair<const component_type, int>>>;
    using pair_type = std::pair<component_type, int>;

    map_type comp_freq(NUM_SAMPLES);
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<GNode> dist(0, graph.size() - 1);
    for (GNode i = 0; i < NUM_SAMPLES; i++) {
      NodeData& ndata = graph.getData(dist(rng), galois::MethodFlag::UNPROTECTED);
      comp_freq[ndata.component()]++;
    }

    assert(!comp_freq.empty());
    auto most_frequent = std::max_element(comp_freq.begin(), comp_freq.end(),
      [](const pair_type& a, const pair_type& b) { return a.second < b.second; });

    galois::gDebug("Approximate largest intermediate component: ", most_frequent->first,
      " (hit rate ", 100.0*(most_frequent->second) / NUM_SAMPLES, "%)");

    return most_frequent->first;
  }

  const uint32_t NEIGHBOR_ROUNDS = 2;
  void operator()(Graph& graph) {
    // (bozhi) should NOT go through single direction in sampling step: nodes with edges less than NEIGHBOR_ROUNDS will fail
    for (uint32_t r = 0; r < NEIGHBOR_ROUNDS; ++r) {
      galois::do_all(
        galois::iterate(graph),
        [&](const GNode& src) {
          Graph::edge_iterator ii =
              graph.edge_begin(src, galois::MethodFlag::UNPROTECTED);
          Graph::edge_iterator ei =
              graph.edge_end(src, galois::MethodFlag::UNPROTECTED);
          for (std::advance(ii, r); ii < ei; ii++) {
            GNode dst = graph.getEdgeDst(ii);
            NodeData& sdata = graph.getData(src, galois::MethodFlag::UNPROTECTED);
            NodeData& ddata = graph.getData(dst, galois::MethodFlag::UNPROTECTED);
            sdata.link(&ddata);
            break;
          }
        },
        galois::steal(),
        galois::loopname("Sample"));

      galois::do_all(
        galois::iterate(graph),
        [&](const GNode& src) {
          NodeData& sdata = graph.getData(src, galois::MethodFlag::UNPROTECTED);
          sdata.compress();
        },
        galois::steal(),
        galois::loopname("Compress_0")
      );
    }

    galois::StatTimer StatTimer_Sampling("Sampling");
    StatTimer_Sampling.start();
    const component_type c = approxLargestComponent(graph);
    StatTimer_Sampling.stop();

    galois::do_all(
      galois::iterate(graph),
      [&](const GNode& src) {
        NodeData& sdata = graph.getData(src, galois::MethodFlag::UNPROTECTED);
        if (sdata.component() == c)
          return;
        Graph::edge_iterator ii =
            graph.edge_begin(src, galois::MethodFlag::UNPROTECTED);
        Graph::edge_iterator ei =
            graph.edge_end(src, galois::MethodFlag::UNPROTECTED);
        for (std::advance(ii, NEIGHBOR_ROUNDS); ii < ei; ++ii) {
          GNode dst = graph.getEdgeDst(ii);
          NodeData& ddata = graph.getData(dst, galois::MethodFlag::UNPROTECTED);
          sdata.link(&ddata);
        }
      },
      galois::steal(),
      galois::loopname("Merge"));

    galois::do_all(
      galois::iterate(graph),
      [&](const GNode& src) {
        NodeData& sdata = graph.getData(src, galois::MethodFlag::UNPROTECTED);
        sdata.compress();
      },
      galois::steal(),
      galois::loopname("Compress_1")
    );
  }
};
