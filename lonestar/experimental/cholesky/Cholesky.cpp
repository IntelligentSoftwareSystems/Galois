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

// A bunch of this is copied from SpanningTree

#include "galois/Galois.h"
#include "galois/Reduction.h"
#include "galois/Bag.h"
#include "galois/Timer.h"
#include "galois/UnionFind.h"
#include "galois/graphs/LCGraph.h"
#include "galois/ParallelSTL.h"
#include "galois/graphs/Graph.h" // MorphGraph
#include "llvm/Support/CommandLine.h"
#include <float.h> // For DBL_DIG, significant digits in double

#include "Lonestar/BoilerPlate.h"

#include <list>
#include <utility>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <cstdio> // For certain debugging output

namespace cll = llvm::cl;

const char* name = "Cholesky Factorization";
const char* desc = "Compute the Cholesky factorization of a graph";
const char* url  = NULL;

enum Ordering { sequential, leastdegree, pointless, fileorder };

static cll::opt<std::string>
    inputFilename(cll::Positional, cll::desc("<graph file>"), cll::Required);
static cll::opt<Ordering> ordering(
    "ordering", cll::desc("Graph traversal order:"),
    cll::values(clEnumVal(sequential, "Sequential ordering"),
                clEnumVal(leastdegree, "Least-degree ordering"),
                clEnumVal(pointless, "1,6,4,5,0,3,7,2;9,... ordering"),
                clEnumVal(fileorder, "Read from orderingFile"), clEnumValEnd),
    cll::init(leastdegree));
static cll::opt<std::string>
    orderingFile("orderingFile",
                 cll::desc("read/write ordering from/to this file"));
static cll::opt<bool>
    skipNumeric("skipNumeric",
                cll::desc("do not perform numeric factorization"));

struct Node {
  unsigned int id;
  unsigned int order;
  unsigned int seen;
  unsigned int noutedges;
  unsigned int degree;
  Node() : order(0), seen(0), noutedges(0), degree(0){};
};

// WARNING: Will silently behave oddly when given a .gr file with the
// wrong data type
typedef double edgedata;
// typedef float edgedata;

// LC_Linear_Graph cannot have structure modified; not suitable for
// symbolic factorization.
// typedef
// galois::graphs::LC_Linear_Graph<Node,edgedata>::with_numa_alloc<true>::type
// Graph;
typedef galois::graphs::MorphGraph<Node, edgedata, true, false, false> Graph;
typedef galois::graphs::MorphGraph<Node, edgedata, false, false, true>
    SymbolicGraph;

typedef Graph::GraphNode GNode;
typedef SymbolicGraph::GraphNode SGNode;

unsigned int nodecount = 0;

std::ostream& operator<<(std::ostream& os, const Node& n) {
  os << "[id: " << &n << "]";
  return os;
}

// Adapted from preflowpush/Preflowpush.cpp
// Find the edge between src and dst. Sets hasEdge if such an edge was found.
template <typename GraphType, typename NodeType>
typename GraphType::edge_iterator findEdge(GraphType& g, NodeType src,
                                           NodeType dst, bool* hasEdge) {
  typename GraphType::edge_iterator ii = g.findEdge(src, dst);
  *hasEdge = ii != g.edge_end(src, galois::MethodFlag::UNPROTECTED);
  return ii;
#if 0
  typename GraphType::edge_iterator
    ii = g.edge_begin(src, galois::MethodFlag::UNPROTECTED),
    ei = g.edge_end(src, galois::MethodFlag::UNPROTECTED),
    origei = ei;
  unsigned targetid = g.getData(dst).id;
  // Binary search
  while ( ii < ei ) {
    typename GraphType::edge_iterator i = ii + (ei - ii)/2;
    NodeType midnode = g.getEdgeDst(i);
    unsigned id = g.getData(dst).id;
    if ( id == targetid ) {
      *hasEdge = true;
      return ii;
    }
    else if ( id < targetid )
      ii = i+1;
    else if ( id > targetid )
      ei = i;
  }
  *hasEdge = false;
  return origei;
  /*
  for (; ii != ei; ++ii) {
    if (g.getEdgeDst(ii) == dst) {
      *hasEdge = true;
      break;
    }
  }
  return ii;
  */
#endif
}

// include/galois/graphs/Serialize.h
// Output a graph to a file as an edgelist
template <typename GraphType>
bool outputTextEdgeData(const char* ofile, GraphType& G) {
  // std::ofstream file(ofile);
  FILE* file = fopen(ofile, "w");
  if (!file) {
    perror("fopen outfile");
    return false;
  }
  for (typename GraphType::iterator ii = G.begin(), ee = G.end(); ii != ee;
       ++ii) {
    unsigned src = G.getData(*ii).id;
    // FIXME: Version in include/galois/graphs/Serialize.h is wrong.
    for (typename GraphType::edge_iterator jj = G.edge_begin(*ii),
                                           ej = G.edge_end(*ii);
         jj != ej; ++jj) {
      unsigned dst = G.getData(G.getEdgeDst(jj)).id;
      fprintf(file, "%d %d %.*e\n", src, dst, DBL_DIG + 3, G.getEdgeData(jj));
    }
  }
  fclose(file);
  return true;
}

struct OrderingManager {
  SymbolicGraph& graph;
  std::vector<SGNode>& innodes;
  OrderingManager(SymbolicGraph& graph, std::vector<SGNode>& innodes)
      : graph(graph), innodes(innodes) {}

  std::ifstream orderfile;

  // Initialize data required to implement the given ordering
  void init() {
    switch (ordering) {
    case leastdegree:
      // Least-degree ordering: count each node's degree
      // FIXME: parallelize?
      for (SymbolicGraph::iterator ii = graph.begin(), ei = graph.end();
           ii != ei; ++ii) {
        SGNode node         = *ii;
        unsigned int degree = 0;
        // Measure degree of the node
        for (SymbolicGraph::edge_iterator
                 iis = graph.edge_begin(node, galois::MethodFlag::WRITE),
                 eis = graph.edge_end(node, galois::MethodFlag::WRITE);
             iis != eis; ++iis)
          degree++;
        assert(degree > 0);
        Node& noded = graph.getData(node);
        assert(noded.degree == 0);
        noded.degree = degree;
        leastdegree_list.push_back(noded.id);
        // leastdegree_update_old(noded.id, true);
      }
      leastdegree_sort(); // One-time cost; shouldn't matter for large matrices
      break;
    case fileorder:
      if (!orderingFile.c_str()[0]) {
        std::cerr << "fileorder: No ordering file specified.\n";
        assert(false && "No ordering file specified.");
        abort();
      }
      orderfile.open(orderingFile.c_str());
      if (!orderfile.good()) {
        std::cerr << "fileorder: Error opening ordering file for read.\n";
        assert(false && "Error opening ordering file for read.\n");
        abort();
      }
      break;
    default:
      break;
    }
  }

  typedef std::list<unsigned int> ldlist;
  ldlist leastdegree_list;

  bool leastdegree_list_compare(unsigned int id1, unsigned int id2) {
    assert(id1 < nodecount && id2 < nodecount);
    unsigned int d1 = graph.getData(innodes[id1]).degree,
                 d2 = graph.getData(innodes[id2])
                          .degree; // FIXME: Does this work?
    return d1 == d2 ? id1 < id2 : d1 < d2;
  }

  // Find the unseen node in the graph of least degree
  unsigned int leastdegree_impl(unsigned int i, unsigned int seenbase = 0) {
    unsigned int nseen = 0, bestid = 0, bestdegree = nodecount + 1;
    // Iterate over nodes
    for (SymbolicGraph::iterator ii = graph.begin(), ei = graph.end(); ii != ei;
         ++ii) {
      SGNode node = *ii;
      Node& noded = graph.getData(node);
      if (noded.seen > seenbase) {
        nseen++;
        continue;
      }
      if (noded.degree < bestdegree) {
        bestid     = noded.id;
        bestdegree = noded.degree;
        // We can't do better than 0
        if (bestdegree == 0)
          break;
      }
    }
    assert(nseen == i || bestdegree == 0);
    return bestid;
  }

#if 0
  void leastdegree_verify_sorted() {
    unsigned int last = 0;
    bool first = true;
    for ( ldlist::iterator ii = leastdegree_list.begin(),
            ei = leastdegree_list.end(); ii != ei; ii++ ) {
      if ( !first )
        assert(leastdegree_list_compare(last, *ii));
      last = *ii;
      first = false;
    }
  }
#endif

  void leastdegree_update_old(unsigned int id, bool commit) {
    assert(ordering == leastdegree);
    leastdegree_list.remove(id);
    // We might make multiple changes to the degree. We need to keep
    // the list sorted, so we remove this node from the list right
    // away. We only re-add the node after the last change.
    if (!commit)
      return;
    ldlist newlist(1, id);
    // Evil hack: http://stackoverflow.com/a/15841503
    // leastdegree_verify_sorted();
    leastdegree_list.merge(newlist, [=](unsigned int id1, unsigned int id2) {
      return this->leastdegree_list_compare(id1, id2);
    });
    // leastdegree_verify_sorted();
    // FIXME: This is not very efficient. We shouldn't have to move the node
    // very far in this (potentially long) list.
  }

  void leastdegree_update(unsigned int id, bool commit) {
    // leastdegree_update_old(id, commit);
    // return;
    galois::StatTimer TX("TimeSymbolicLeastDegreeUpdate");
    TX.start();
    assert(ordering == leastdegree);
    ldlist::iterator ii = leastdegree_list.begin(), bi = ii,
                     ei = leastdegree_list.end();
    // Locate the item we need to move
    for (;; ii++) {
      if (ii == ei) {
        assert(false && "leastdegree_update overflow");
        abort();
      }
      if (*ii == id)
        break;
    }
    // Save item position and determine next node and previous node.
    ldlist::iterator item = ii;
    ii++;
    ldlist::iterator item_next = ii;
    while (ii != ei && leastdegree_list_compare(*ii, *item))
      ii++;
    if (ii == item_next) {
      // We don't want to move it further, it's in the correct
      // position relative to the next item. Maybe we need to move it
      // back.
      ii                           = item;
      ldlist::iterator item_actual = ii;
      item_actual++;
      while (!leastdegree_list_compare(*ii, *item)) {
        item_actual--;
        if (ii == bi)
          break;
        else
          ii--;
      }
      ii = item_actual;
    }
    TX.stop();
    // If we still haven't moved, we don't need to move the node
    if (ii == item_next)
      return;
    // Move the item to be before ii.
    leastdegree_list.splice(ii, leastdegree_list, item);
  }

  void leastdegree_sort() {
    // Evil hack: http://stackoverflow.com/a/15841503
    leastdegree_list.sort([=](unsigned int id1, unsigned int id2) {
      return this->leastdegree_list_compare(id1, id2);
    });
  }

  // For the given ordering, return the ID of the next node that should
  // be eliminated.
  unsigned int next_node(unsigned int i, unsigned int seenbase = 0) {
    static const unsigned int pointless_len    = 8,
                              pointless_data[] = {
                                  1, 6, 4, 5,
                                  0, 3, 7, 2}; // For "pointless" ordering
    assert(i < nodecount);
    unsigned int result = 0;
    bool overflow       = true;

    switch (ordering) {
    case sequential:
      result = i;
      break;
    case leastdegree:
      // leastdegree_update3();
      // result = leastdegree_impl(i, seenbase);
      result = leastdegree_list.front();
      leastdegree_list.pop_front();
      break;
    case pointless:
      for (unsigned int offset = i % pointless_len, base = i - offset, j = 0;
           j < pointless_len; j++) {
        unsigned int pointless_result = base + pointless_data[j];
        if (pointless_result >= nodecount)
          continue;
        if (offset == 0) {
          result   = pointless_result;
          overflow = false;
          break;
        }
        offset--;
      }
      assert(!overflow && "Pointless overflow");
      break;
    case fileorder:
      orderfile >> result;
      if (!orderfile.good()) {
        std::cerr << "Ordering file read error\n";
        abort();
      }
      if (i == nodecount) {
        std::cout << "Closing ordering file\n";
        orderfile.close();
      }
      break;
    default:
      std::cerr << "Unknown ordering: " << ordering << "\n";
      abort();
    }
    // printf("%d %d\n", result, graph.getData(innodes[result]).degree-1);
    return result;
  }
};

/**
 * Perform the symbolic factorization. Modifies the graph structure.
 * Produces as output a (directed) graph to use with NumericAlgo, the
 * numeric factorization.
 */
template <typename GraphType, typename OutGraphType>
struct SymbolicAlgo {
  GraphType& graph;
  OutGraphType& outgraph;

  std::vector<GNode> outnodes;
  std::vector<SGNode> innodes;
  OrderingManager ordermgr;

  SymbolicAlgo(GraphType& graph, OutGraphType& outgraph)
      : graph(graph), outgraph(outgraph), ordermgr(graph, innodes){};

  template <typename C>
  void operator()(SGNode node, C& ctx) {
    // Update seen flag on node
    Node& noded = graph.getData(node);
    assert(noded.seen == 0);
    noded.seen = 1;
    // FIXME: Be "cautious"

    // Make sure remaining neighbors form a clique
    // It should be safe to add edges between neighbors here.
    for (typename GraphType::edge_iterator
             iis = graph.edge_begin(node, galois::MethodFlag::WRITE),
             eis = graph.edge_end(node, galois::MethodFlag::WRITE);
         iis != eis; ++iis) {
      SGNode src = graph.getEdgeDst(iis);
      Node& srcd = graph.getData(src);
      if (srcd.seen > 0)
        continue;

      // Enumerate all other neighbors
      for (typename GraphType::edge_iterator
               iid = graph.edge_begin(node, galois::MethodFlag::WRITE),
               eid = graph.edge_end(node, galois::MethodFlag::WRITE);
           iid != eid; ++iid) {
        SGNode dst = graph.getEdgeDst(iid);
        Node& dstd = graph.getData(dst);
        if (dstd.seen > 0)
          continue;

        // Find the edge that bridges these two neighbors
        bool hasEdge = false;
        typename GraphType::edge_iterator bridge =
            findEdge(graph, src, dst, &hasEdge);
        if (hasEdge)
          continue;

        // The edge doesn't exist, so add an undirected edge between
        // these two nodes
        bridge = graph.addEdge(src, dst, galois::MethodFlag::WRITE);
        edgedata& ed =
            graph.getEdgeData(bridge, galois::MethodFlag::UNPROTECTED);
        ed = 0;
        if (ordering == leastdegree) {
          srcd.degree++;
          dstd.degree++;
          ordermgr.leastdegree_update(srcd.id, false);
          ordermgr.leastdegree_update(dstd.id, false);
        }
      }
    }

    // std::cout << "Counting edges for node " << noded.id << ": ";

    // Update degree for this node (zero because it's eliminated)
    noded.degree = 0;

    // Count number of edges to add to the output graph. These will be
    // preallocated and added later.
    for (typename GraphType::edge_iterator
             iis = graph.edge_begin(node, galois::MethodFlag::WRITE),
             eis = graph.edge_end(node, galois::MethodFlag::WRITE);
         iis != eis; ++iis) {
      SGNode src = graph.getEdgeDst(iis);
      Node& srcd = graph.getData(src);
      if (srcd.seen == 0) {
        // We're eliminating ourself, so update the degree of this
        // unseen neighbor
        if (ordering == leastdegree) {
          srcd.degree--;
          ordermgr.leastdegree_update(srcd.id, true);
        }
      } else {
        srcd.noutedges++;
        assert(srcd.degree == 0);
        // std::cout << "(" << srcd.id << ")";
      }
    }
    // std::cout << "\n";
  }

  int nzerodeg = 0;

  void add_outedges(SGNode node) { // FIXME: &
    Node& noded = graph.getData(node);
    assert(noded.seen == 1);
    noded.seen    = 2;
    bool doneself = false;

    // std::cout << "Adding edges targeting node " << noded.id << " ";

    // Undirected graph double-counts self-edge
    assert(noded.noutedges > 1);
    noded.noutedges--;
    // std::cout << "[a source of " << noded.noutedges << " edges]: ";
    // Create our node and allocate our edges
    GNode outnode = outgraph.createNode(noded);
    outgraph.addNode(outnode);
    outgraph.resizeEdges(outnode, noded.noutedges);
    outnodes[noded.id] = outnode;
    assert(outgraph.getData(outnode).id == noded.id);
    assert(outgraph.getData(outnode).seen == 2);

    // Add edges to the output (elimination graph).
    int indegree = 0;
    for (typename GraphType::edge_iterator
             iis = graph.edge_begin(node, galois::MethodFlag::WRITE),
             eis = graph.edge_end(node, galois::MethodFlag::WRITE);
         iis != eis; ++iis) {
      SGNode src = graph.getEdgeDst(iis);
      Node& srcd = graph.getData(src);
      // std::cout << "(" << srcd.id << ")";
      if (srcd.seen == 1)
        continue; // 1 = not seen; 2 = seen
      if (srcd.id == noded.id) {
        if (doneself)
          continue;
        doneself = true;
      } else
        indegree++;
      assert(srcd.noutedges > 0);
      // std::cout << "Y ";
      // Add a directed edge from src to node (copying weight)
      typename OutGraphType::edge_iterator edge = outgraph.addEdge(
          outnodes[srcd.id], outnode, galois::MethodFlag::WRITE);
      edgedata& ed = outgraph.getEdgeData(edge);
      ed           = graph.getEdgeData(iis);
      // Bookkeeping
      srcd.noutedges--;
    }
    if (indegree == 0)
      nzerodeg++;
    // std::cout << "\n";
  }

  void operator()() {
    // Initialize the output (directed) graph: create nodes
    unsigned int nodeorder[nodecount];
    unsigned int nodeID = 0;

    outnodes.resize(nodecount);
    innodes.resize(nodecount);
    for (typename GraphType::iterator ii = graph.begin(), ei = graph.end();
         ii != ei; ++ii) {
      innodes[nodeID] = *ii;
      nodeID++;
    }
    assert(nodeID == nodecount);
    galois::StatTimer TI("TimeSymbolicInit");
    TI.start();
    ordermgr.init();
    TI.stop();

    // Eliminate each node in given traversal order.
    // FIXME: parallelize? See paper.
    galois::StatTimer TX("TimeSymbolicX");
    for (unsigned int i = 0; i < nodecount; i++) {
      // Append next node to execution order
      nodeorder[i] = ordermgr.next_node(i);
      // std::cout << "Eliminating " << i << "\n";
      SGNode node               = innodes[nodeorder[i]];
      graph.getData(node).order = i;
      void* emptyctx            = NULL;
      TX.start();
      (*this)(node, emptyctx);
      TX.stop();
    }

    // Consider writing the ordering file
    if (ordering != fileorder && orderingFile.c_str()[0]) {
      std::ofstream orderfile;
      std::cout << "Saving ordering to " << orderingFile.c_str() << "\n";
      orderfile.open(orderingFile.c_str());
      if (!orderfile.good()) {
        std::cerr << "fileorder: Error opening ordering file for write.\n";
        assert(false && "Error opening ordering file for write.\n");
        abort();
      }
      for (unsigned int i = 0; i < nodecount; i++)
        orderfile << nodeorder[i] << "\n";
      if (!orderfile.good()) {
        std::cerr << "Ordering file write error\n";
        abort();
      }
      orderfile.close();
    }

    // Verify that all nodes have been eliminated before building outgraph
    for (unsigned int i = 0; i < nodecount; i++)
      assert(graph.getData(innodes[i]).seen == 1);
    // Preallocate edges and add them to the output graph
    for (unsigned int i = 0; i < nodecount; i++)
      add_outedges(innodes[nodeorder[i]]);
    printf("%d nodes with zero in-degree.\n", nzerodeg);
    // Verify that the correct number of edges were added
    for (unsigned int i = 0; i < nodecount; i++)
      assert(graph.getData(innodes[i]).noutedges == 0);
  }
};

/**
 * Comparison function. The symbolic factorization produces a total
 * ordering of the nodes. In conjunction with the neighborhood
 * function, this defines the traversal order for the numeric
 * factorization.
 */
template <typename GraphType>
struct Cmp {
  GraphType& graph;
  Cmp(GraphType& graph) : graph(graph){};

  bool operator()(const GNode& node1, const GNode& node2) const {
    Node& node1d = graph.getData(node1, galois::MethodFlag::UNPROTECTED);
    Node& node2d = graph.getData(node2, galois::MethodFlag::UNPROTECTED);
    bool result  = node1d.order <= node2d.order;
    /*
    std::cout << "Cmp: " << node1d.id << " <= " << node2d.id << ": " <<
      (result ? "true" : "false") << "\n";
    */
    return result;
  }
};

/**
 * Defining the neighborhood of the operator. The operator touches all
 * of the edges to and between neighbors. Nodes with overlapping
 * neighborhoods won't be executed in parallel.
 */
template <typename GraphType>
struct NhFunc {
  /*
  // Affect for_each_ordered's choice of executor. This has certain issues.
  typedef int tt_has_fixed_neighborhood;
  static_assert(galois::has_fixed_neighborhood<NhFunc>::value, "Oops!");
  */

  GraphType& graph;
  NhFunc(GraphType& graph) : graph(graph){};

  template <typename C>
  void operator()(GNode& node, C& ctx) {
    (*this)(node);
  }
  void operator()(GNode& node) {
    // Touch all neighbors (this seems to be good enough)
    Graph::edge_iterator ii = graph.edge_begin(node, galois::MethodFlag::WRITE);
  }
};

/**
 * Perform the numeric factorization. Assumes the graph is a directed
 * graph as produced by the symbolic factorization.
 */
template <typename GraphType>
struct NumericAlgo {
  /*
  // Affect for_each_ordered's choice of executor. This has certain issues.
  typedef int tt_does_not_need_push;
  static_assert(galois::does_not_need_push<NumericAlgo>::value, "Oops!");
  */

  GraphType& graph;
  NumericAlgo(GraphType& graph) : graph(graph){};

  void operator()(GNode node, galois::UserContext<GNode>& ctx) {
    // Update seen flag on node
#define locktype galois::MethodFlag::UNPROTECTED
    Node& noded = graph.getData(node);
    assert(noded.seen == 0);
    noded.seen = 1;

    // std::cout << "STARTING " << noded.id << "\n";

    // Find self-edge for this node, update it
    bool hasEdge = false;
    edgedata& factor =
        graph.getEdgeData(findEdge(graph, node, node, &hasEdge), locktype);
    assert(hasEdge);
    assert(factor > 0);
    factor = sqrt(factor);
    assert(factor != 0 && !isnan(factor));

    // std::cout << "STARTING " << noded.id << " " << factor << "\n";
    // printf("STARTING %4d %10.5f\n", noded.id, factor);

    // Update all edges (except self-edge)
    for (Graph::edge_iterator ii = graph.edge_begin(node, locktype),
                              ei = graph.edge_end(node, locktype);
         ii != ei; ++ii) {
      GNode dst  = graph.getEdgeDst(ii);
      Node& dstd = graph.getData(dst);
      if (dstd.seen == 0) {
        edgedata& ed = graph.getEdgeData(ii, locktype);
        ed /= factor;
        // printf("N-EDGE %4d %4d %10.5f\n", noded.id, graph.getData(dst).id,
        // ed); std::cout << noded.id << " " << dstd.id << " " << ed << "\n";
      }
    }

    // Update all edges between neighbors (we're operating on the filled graph,
    // so we they form a (directed) clique)
    for (Graph::edge_iterator iis = graph.edge_begin(node, locktype),
                              eis = graph.edge_end(node, locktype);
         iis != eis; ++iis) {
      GNode src  = graph.getEdgeDst(iis);
      Node& srcd = graph.getData(src);
      if (srcd.seen > 0)
        continue;
      edgedata& eds = graph.getEdgeData(iis, locktype);

      // Enumerate all other neighbors
      for (Graph::edge_iterator iid = graph.edge_begin(node, locktype),
                                eid = graph.edge_end(node, locktype);
           iid != eid; ++iid) {
        GNode dst  = graph.getEdgeDst(iid);
        Node& dstd = graph.getData(dst);
        if (dstd.seen > 0)
          continue;

        // Find the edge that bridges these two neighbors
        hasEdge                     = false;
        Graph::edge_iterator bridge = findEdge(graph, src, dst, &hasEdge);
        if (!hasEdge)
          continue;

        // Update the weight of the bridge edge
        edgedata &edd = graph.getEdgeData(iid, locktype),
                 &edb = graph.getEdgeData(bridge, locktype);
        edb -= eds * edd;

        // printf("I-EDGE %4d %4d %10.5f\n", srcd.id, dstd.id, edb);
        // std::cout << srcd.id << " " << dstd.id << " " << edb << "\n";
      }
    }
    // std::cout << "OPERATED ON " << noded.id << "\n";
    // sleep(1); // Maybe use this to help debug parallelism
  }

  void operator()() {
    Graph::iterator ii = graph.begin(), ei = graph.end();
    if (ii == ei) {
      assert(false && "Empty matrix?");
      abort();
    }
    galois::for_each_ordered(ii, ei, Cmp<GraphType>(graph),
                             NhFunc<GraphType>(graph), *this);
    // galois::for_each(ii, ei, *this);
  }
};

// Load a graph into a MorphGraph. Based on makeGraph from Boruvka.
template <typename GraphType>
static void makeGraph(GraphType& graph, const char* input) {
  std::vector<SGNode> nodes;
  // Create local computation graph.
  typedef galois::graphs::LC_CSR_Graph<Node, edgedata> InGraph;
  typedef InGraph::GraphNode InGNode;
  InGraph in_graph;
  // Read graph from file.
  galois::graphs::readGraph(in_graph, input);
  std::cout << "Read " << in_graph.size() << " nodes\n";
  // A node and a int is an element.
  typedef std::pair<InGNode, edgedata> Element;
  // A vector of element is 'Elements'
  typedef std::vector<Element> Elements;
  // A vector of 'Elements' is a 'Map'
  typedef std::vector<Elements> Map;
  //'in_edges' is a vector of vector of pairs of nodes and int.
  Map edges(in_graph.size());
  //
  int numEdges = 0;
  // Extract edges from input graph
  for (InGraph::iterator src = in_graph.begin(), esrc = in_graph.end();
       src != esrc; ++src) {
    for (InGraph::edge_iterator
             dst  = in_graph.edge_begin(*src, galois::MethodFlag::UNPROTECTED),
             edst = in_graph.edge_end(*src, galois::MethodFlag::UNPROTECTED);
         dst != edst; ++dst) {
      edgedata w = in_graph.getEdgeData(dst);
      Element e(*src, w);
      edges[in_graph.getEdgeDst(dst)].push_back(e);
      numEdges++;
    }
  }
  //#if BORUVKA_DEBUG
  std::cout << "Number of edges " << numEdges << std::endl;
  //#endif
  // Create nodes in output graph
  nodes.resize(in_graph.size());
  int nodeID = 0;
  for (Map::iterator i = edges.begin(), ei = edges.end(); i != ei; ++i) {
    Node n;
    n.id = nodeID;
    assert(n.seen == 0);
    SGNode node = graph.createNode(n);
    graph.addNode(node);
    nodes[nodeID] = node;
    nodeID++;
  }

  int id   = 0;
  numEdges = 0;
  for (Map::iterator i = edges.begin(), ei = edges.end(); i != ei; ++i) {
    SGNode src = nodes[id];
    for (Elements::iterator j = i->begin(), ej = i->end(); j != ej; ++j) {
      typename GraphType::edge_iterator it =
          graph.findEdge(src, nodes[j->first], galois::MethodFlag::UNPROTECTED);
      if (it != graph.edge_end(src, galois::MethodFlag::UNPROTECTED)) {
        assert(graph.getEdgeData(it) == j->second);
        continue;
      }
      it = graph.addEdge(src, nodes[j->first], galois::MethodFlag::UNPROTECTED);
      graph.getEdgeData(it) = j->second;
      numEdges++;
    }
    id++;
  }
  //#if BORUVKA_DEBUG
  std::cout << "Final num edges " << numEdges << std::endl;
  //#endif
}

// FIXME: implement verify, etc. See SpanningTree.

template <typename GraphType>
bool verify(GraphType& graph) {
  outputTextEdgeData("choleskyedges.txt", graph);
  std::cout << "\n\n\nPlease verify by comparing ./choleskyedges.txt against "
               "expected contents.\n\n\n\n";
  // FIXME: Try multiplying to double-check result
  return true;
  /*
  if (galois::ParallelSTL::find_if(graph.begin(), graph.end(), is_bad_graph())
  == graph.end()) { if (galois::ParallelSTL::find_if(mst.begin(), mst.end(),
  is_bad_mst()) == mst.end()) { CheckAcyclic c; return c();
    }
  }
  return false;
  */
}

template <typename Algo>
void run(Algo& algo, const char* algoname) {
  galois::StatTimer T, U(algoname);
  T.start();
  U.start();
  algo();
  T.stop();
  U.stop();
}

int main(int argc, char** argv) {
  galois::StatManager statManager;
  LonestarStart(argc, argv, name, desc, url);

  galois::StatTimer Tinitial("InitializeTime");
  Tinitial.start();

  SymbolicGraph graph;
  Graph outgraph;
  unsigned int edgecount = 0;

  // Load input graph. Read to an LC_Graph and then convert to a
  // MorphGraph. (based on makeGraph from Boruvka.)
  makeGraph(graph, inputFilename.c_str());
  nodecount = graph.size();
  std::cout << "Num nodes: " << nodecount << "\n";

  // Verify IDs assigned to each node
  {
    unsigned int i = 0;
    for (SymbolicGraph::iterator ii = graph.begin(), ei = graph.end(); ii != ei;
         ++ii) {
      Node& data = graph.getData(*ii);
      assert(data.id == i++);
      assert(data.seen == 0);
      edgecount++;
      for (SymbolicGraph::edge_iterator
               iid = graph.edge_begin(*ii, galois::MethodFlag::WRITE),
               eid = graph.edge_end(*ii, galois::MethodFlag::WRITE);
           iid != eid; ++iid)
        if (data.id < graph.getData(graph.getEdgeDst(iid)).id)
          edgecount++;
    }
    assert(i == nodecount);
  }

  Tinitial.stop();

  // galois::preAlloc(numThreads);
  galois::reportPageAlloc("MeminfoPre");

  // First run the symbolic factorization
  std::cout << "Symbolic factorization\n";
  {
    SymbolicAlgo<SymbolicGraph, Graph> algo(graph, outgraph);
    run(algo, "TimeSymbolic");
  }

  // Clear the seen flags for the numeric factorization.
  unsigned int newedgecount = 0;
  for (Graph::iterator ii = outgraph.begin(), ei = outgraph.end(); ii != ei;
       ++ii) {
    Node& data = outgraph.getData(*ii);
    assert(data.seen == 2);
    data.seen = 0;
    for (Graph::edge_iterator
             iid = outgraph.edge_begin(*ii, galois::MethodFlag::WRITE),
             eid = outgraph.edge_end(*ii, galois::MethodFlag::WRITE);
         iid != eid; ++iid)
      newedgecount++;
  }
  assert(newedgecount >= edgecount);
  std::cout << "Added " << (newedgecount - edgecount) << " edges\n";

  if (!skipVerify)
    outputTextEdgeData("fillededges.txt", outgraph);
  if (skipNumeric)
    return 0;

  // We should now have built a directed graph (outgraph) and total
  // ordering. Now run the numeric factorization.
  //
  // FIXME: Convert back to a LC_Graph?
  std::cout << "Numeric factorization\n";
  {
    NumericAlgo<Graph> algo(outgraph);
    run(algo, "TimeNumeric");
  }

  galois::reportPageAlloc("MeminfoPost");

  if (!skipVerify && !verify(outgraph)) {
    std::cerr << "verification failed\n";
    assert(0 && "verification failed");
    abort();
  }

  return 0;
}
