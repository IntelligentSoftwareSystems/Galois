/** Write graphs out to file -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
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
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */
#ifndef GALOIS_GRAPHS_SERIALIZE_H
#define GALOIS_GRAPHS_SERIALIZE_H

#include <sys/stat.h>
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>

#include <fstream>
#include <algorithm>

namespace Galois {
namespace Graph {

template<typename Graph>
struct CompareNodeData {
  typedef typename Graph::GraphNode GNode;
  bool operator()(const GNode& a, const GNode& b) {
    return a.getData() < b.getData();
  }
};

/**
 * Writes graph out to binary file. Does not currently save node
 * data. Does try to make node ids stable by sorting Graph::GraphNodes
 * using NodeData::operator<() first, so if you want nodes to appear
 * in certain order set the node data appropriately.
 */
template<typename Graph>
bool outputGraph(const char* file, Graph& G) {
  ssize_t retval;
  //ASSUME LE machine
  mode_t mode = S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH;
  int fd = open(file, O_WRONLY | O_CREAT |O_TRUNC, mode);
  
  //version
  uint64_t version = 1;
  retval = write(fd, &version, sizeof(uint64_t));
  if (retval == -1) {
    perror(__FILE__);
    abort();
  }

  uint64_t sizeof_edge_data = sizeof(typename Graph::EdgeDataTy);
  retval = write(fd, &sizeof_edge_data, sizeof(uint64_t));
  if (retval == -1) {
    perror(__FILE__);
    abort();
  }

  //num nodes
  uint64_t num_nodes = G.size();
  retval = write(fd, &num_nodes, sizeof(uint64_t));
  if (retval == -1) {
    perror(__FILE__);
    abort();
  }

  typedef typename Graph::GraphNode GNode;
  typedef std::vector<GNode> Nodes;
  Nodes nodes;
  for (typename Graph::iterator ii = G.begin(),
      ee = G.end(); ii != ee; ++ii) {
    nodes.push_back(*ii);
  }

  // TODO(ddn): for some reason stable_sort crashes with:
  //  free(): invalid pointer
  std::sort(nodes.begin(), nodes.end(), CompareNodeData<Graph>());

  //num edges and outidx computation
  uint64_t offset = 0;
  std::vector<uint64_t> out_idx;
  std::map<typename Graph::GraphNode, uint32_t> node_ids;
  for (uint32_t id = 0; id < num_nodes; ++id) {
    GNode& node = nodes[id];
    node_ids[node] = id;
    offset += G.neighborsSize(node);
    out_idx.push_back(offset);
  }
  retval = write(fd, &offset, sizeof(uint64_t));
  if (retval == -1) {
    perror(__FILE__);
    abort();
  }

  //outIdx
  retval = write(fd, &out_idx[0], sizeof(uint64_t) * out_idx.size());
  if (retval == -1) {
    perror(__FILE__);
    abort();
  }

  //outs
  size_t num_edges = 0;
  for (typename Nodes::iterator ii = nodes.begin(), ee = nodes.end();
      ii != ee; ++ii) {
    for (typename Graph::neighbor_iterator ni = G.neighbor_begin(*ii),
        ne = G.neighbor_end(*ii); ni != ne; ++ni, ++num_edges) {
      uint32_t id = node_ids[*ni];
      retval = write(fd, &id, sizeof(uint32_t));
      if (retval == -1) {
        perror(__FILE__);
        abort();
      }
    }
  }
  if (num_edges % 2) {
    uint32_t padding = 0;
    retval = write(fd, &padding, sizeof(uint32_t));
    if (retval == -1) {
      perror(__FILE__);
      abort();
    }
  }

  //edgeData
  for (typename Nodes::iterator ii = nodes.begin(), ee = nodes.end();
      ii != ee; ++ii) {
    for (typename Graph::neighbor_iterator ni = G.neighbor_begin(*ii),
        ne = G.neighbor_end(*ii); ni != ne; ++ni) {
      retval = write(fd, &G.getEdgeData(*ii, *ni),
          sizeof(typename Graph::EdgeDataTy));
      if (retval == -1) {
        perror(__FILE__);
        abort();
      }
    }
  }

  close(fd);

  return true;
}

//! Writes graph out to an ASCII file
template<typename Graph>
bool outputTextEdgeData(const char* ofile, Graph& G) {
  std::ofstream file(ofile);
  for (typename Graph::iterator ii = G.begin(),
	 ee = G.end(); ii != ee; ++ii) {
    for (typename Graph::neighbor_iterator ni = G.neighbor_begin(*ii),
	   ne = G.neighbor_end(*ii); ni != ne; ++ni) {
      file << G.getEdgeData(*ii, *ni) << '\n';
    }
  }
  return true;
}


}
}
#endif
