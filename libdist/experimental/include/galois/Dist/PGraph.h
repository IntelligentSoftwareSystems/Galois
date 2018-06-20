/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
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

/*
 * PGraph.h
 *
 *  Created on: Aug 10, 2015
 *      Author: rashid
 */
#include "galois/Galois.h"
#include "galois/gstl.h"
#include "galois/graphs/FileGraph.h"
#include "galois/graphs/LC_CSR_Graph.h"
#include "galois/graphs/Util.h"
#include "Lonestar/BoilerPlate.h"
#include "galois/Bag.h"
#include "galois/runtime/Network.h"

#ifndef GDIST_EXP_APPS_HPR_PGRAPH_H_
#define GDIST_EXP_APPS_HPR_PGRAPH_H_

/*********************************************************************************
 * Partitioned graph structure.
 **********************************************************************************/
template <typename GraphTy>
struct pGraph {
  typedef typename GraphTy::GraphNode GraphNode;
  typedef typename GraphTy::edge_data_type EdgeDataType;
  typedef typename GraphTy::node_data_type NodeDataType;

  GraphTy g;
  unsigned g_offset; // LID + g_offset = GID
  unsigned numOwned; // [0, numOwned) = global nodes owned, thus [numOwned,
                     // numNodes) are replicas
  unsigned numNodes; // number of nodes (may differ from g.size() to simplify
                     // loading)
  unsigned numEdges;

  // [numNodes, g.size()) should be ignored
  std::vector<unsigned> L2G;        // GID = L2G[LID - numOwned]
  unsigned id;                      // my hostid
  std::vector<unsigned> lastNodes;  //[ [i - 1], [i]) -> Node owned by host i
  unsigned getHost(unsigned node) { // node is GID
    return std::distance(
        lastNodes.begin(),
        std::upper_bound(lastNodes.begin(), lastNodes.end(), node));
  }
  unsigned G2L(unsigned GID) {
    if (GID >= g_offset && GID < g_offset + numOwned)
      return GID - g_offset;
    auto ii = std::find(L2G.begin(), L2G.end(), GID);
    assert(ii != L2G.end());
    return std::distance(L2G.begin(), ii) + numOwned;
  }
  unsigned uid(unsigned lid) {
    assert(lid < numNodes);
    if (lid < numOwned) {
      return lid + g_offset;
    } else {
      return L2G[lid - numOwned];
    }
  }

  pGraph() : g_offset(0), numOwned(0), numNodes(0), id(0), numEdges(0) {}
  /*********************************************************************************
   * Given a partitioned graph  .
   * lastNodes maintains indices of nodes for each co-host. This is computed by
   * determining the number of nodes for each partition in 'pernum', and going
   *over all the nodes assigning the next 'pernum' nodes to the ith partition.
   * The lastNodes is used to find the host by a binary search.
   **********************************************************************************/

  void loadLastNodes(size_t size, unsigned numHosts) {
    if (numHosts == 1)
      return;

    auto p          = galois::block_range(0UL, size, 0, numHosts);
    unsigned pernum = p.second - p.first;
    unsigned pos    = pernum;

    while (pos < size) {
      this->lastNodes.push_back(pos);
      pos += pernum;
    }
#if _HETERO_DEBUG_
    for (int i = 0; size < 10 && i < size; i++) {
      printf("node %d owned by %d\n", i, this->getHost(i));
    }
#endif
  }
  /*********************************************************************************
   * Load a partitioned graph from a file.
   * @param file the graph filename to be loaded.
   * @param hostID the identifier of the current host.
   * @param numHosts the total number of hosts.
   * @param out A graph instance that stores the actual graph.
   * @return a partitioned graph backed by the #out instance.
   **********************************************************************************/
  void loadGraph(std::string file) {
    galois::graphs::FileGraph fg;
    fg.fromFile(file);
    unsigned hostID   = galois::runtime::NetworkInterface::ID;
    unsigned numHosts = galois::runtime::NetworkInterface::Num;
    auto p            = galois::block_range(0UL, fg.size(), hostID, numHosts);
    this->g_offset    = p.first;
    this->numOwned    = p.second - p.first;
    this->id          = hostID;
    std::vector<unsigned> perm(fg.size(), ~0); //[i (orig)] -> j (final)
    unsigned nextSlot = 0;
    //   std::cout << fg.size() << " " << p.first << " " << p.second << "\n";
    // Fill our partition
    for (unsigned i = p.first; i < p.second; ++i) {
      // printf("%d: owned: %d local: %d\n", hostID, i, nextSlot);
      perm[i] = nextSlot++;
    }
    // find ghost cells
    for (auto ii = fg.begin() + p.first; ii != fg.begin() + p.second; ++ii) {
      for (auto jj = fg.edge_begin(*ii); jj != fg.edge_end(*ii); ++jj) {
        // std::cout << *ii << " " << *jj << " " << nextSlot << " " <<
        // perm.size() << "\n";
        //      assert(*jj < perm.size());
        auto dst = fg.getEdgeDst(jj);
        if (perm.at(dst) == ~0) {
          // printf("%d: ghost: %d local: %d\n", hostID, dst, nextSlot);
          perm[dst] = nextSlot++;
          this->L2G.push_back(dst);
        }
      }
    }
    this->numNodes = nextSlot;

    // Fill remainder of graph since permute doesn't support truncating
    for (auto ii = fg.begin(); ii != fg.end(); ++ii)
      if (perm[*ii] == ~0)
        perm[*ii] = nextSlot++;
    //   std::cout << nextSlot << " " << fg.size() << "\n";
    assert(nextSlot == fg.size());
    // permute graph
    galois::graphs::FileGraph fg2;
    galois::graphs::permute<EdgeDataType>(fg, perm, fg2);
    galois::graphs::readGraph(this->g, fg2);

    loadLastNodes(fg.size(), numHosts);

    /* TODO: This still counts edges from ghosts to remote nodes,
     ideally we only want edges from ghosts to local nodes.

     See pGraphToMarshalGraph for one implementation.
     */
    this->numEdges = std::distance(
        this->g.edge_begin(*this->g.begin()),
        this->g.edge_end(*(this->g.begin() + this->numNodes - 1)));
    return;
  }
};
#endif /* GDIST_EXP_APPS_HPR_PGRAPH_H_ */
