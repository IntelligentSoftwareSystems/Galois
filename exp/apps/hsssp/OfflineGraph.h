/** Offline graph -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2014, The University of Texas at Austin. All rights reserved.
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
 * @section Description
 *
 * Offline graph for loading large files
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */

#include <cstdint>
#include <fstream>

#include <boost/iterator/counting_iterator.hpp>

//File format V1:
//version (1) {uint64_t LE}
//EdgeType size {uint64_t LE}
//numNodes {uint64_t LE}
//numEdges {uint64_t LE}
//outindexs[numNodes] {uint64_t LE} (outindex[nodeid] is index of first edge for nodeid + 1 (end interator.  node 0 has an implicit start iterator of 0.
//outedges[numEdges] {uint32_t LE}
//potential padding (32bit max) to Re-Align to 64bits
//EdgeType[numEdges] {EdgeType size}




class OfflineGraph {
  std::ifstream file;
  uint32_t numNodes;
  uint64_t numEdges;
  size_t length;

  uint64_t outIndexs(uint64_t node) {
    file.seekg((4 + node)*sizeof(uint64_t), file.beg);
    uint64_t retval;
    file.read(reinterpret_cast<char*>(&retval), sizeof(uint64_t));
    return retval;
  }

  uint32_t outEdges(uint64_t edge) {
    file.seekg((4 + numNodes) * sizeof(uint64_t) + edge * sizeof(uint32_t), file.beg);
    uint32_t retval;
    file.read(reinterpret_cast<char*>(&retval), sizeof(uint32_t));
    return retval;
  }

public:
  typedef boost::counting_iterator<uint32_t> iterator;
  typedef boost::counting_iterator<uint64_t> edge_iterator;
  typedef uint32_t GraphNode;

  OfflineGraph(const std::string& name)
    :file(name)
  {
    if (!file.is_open() || !file.good()) throw "Bad filename";
    uint64_t ver = 0;
    file.read(reinterpret_cast<char*>(&ver), sizeof(uint64_t));
    file.seekg(sizeof(uint64_t), file.cur);
    file.read(reinterpret_cast<char*>(&numNodes), sizeof(uint64_t));
    file.read(reinterpret_cast<char*>(&numEdges), sizeof(uint64_t));
    if (ver != 1) throw "Bad Version";
    if (!file) throw "Out of data";
    //File length
    file.seekg(0, file.end);
    length = file.tellg();
    if (length < sizeof(uint64_t)*(4+numNodes) + sizeof(uint32_t)*numEdges)
      throw "File too small";
    
  }

  OfflineGraph(OfflineGraph&&) = default;


  size_t size() const { return numNodes; }
  size_t sizeEdges() const { return numEdges; }

  iterator begin() { return iterator(0); }
  iterator end() { return iterator(numNodes); }

  edge_iterator edge_begin(GraphNode N) {
    if (N == 0)
      return edge_iterator(0);
    else
      return edge_iterator(outIndexs(N-1));
  }
  
  edge_iterator edge_end(GraphNode N) {
    return edge_iterator(outIndexs(N));
  }

  GraphNode getEdgeDst(edge_iterator ni) {
    return outEdges(*ni);
  }

};


