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

#include "Galois/Substrate/SimpleLock.h"

#include <cstdint>
#include <fstream>
#include <mutex>

#include <boost/iterator/counting_iterator.hpp>


#ifndef _GALOIS_DIST_OFFLINE_GRAPH_
#define _GALOIS_DIST_OFFLINE_GRAPH_
//File format V1:
//version (1) {uint64_t LE}
//EdgeType size {uint64_t LE}
//numNodes {uint64_t LE}
//numEdges {uint64_t LE}
//outindexs[numNodes] {uint64_t LE} (outindex[nodeid] is index of first edge for nodeid + 1 (end interator.  node 0 has an implicit start iterator of 0.
//outedges[numEdges] {uint32_t LE}
//potential padding (32bit max) to Re-Align to 64bits
//EdgeType[numEdges] {EdgeType size}

//File format V2:
//version (2) {uint64_t LE}
//EdgeType size {uint64_t LE}
//numNodes {uint64_t LE}
//numEdges {uint64_t LE}
//outindexs[numNodes] {uint64_t LE} (outindex[nodeid] is index of first edge for nodeid + 1 (end interator.  node 0 has an implicit start iterator of 0.
//outedges[numEdges] {uint64_t LE}
//EdgeType[numEdges] {EdgeType size}


class OfflineGraph {
  std::ifstream file1;
  std::ifstream fileIndex, fileEdgeDst, fileEdgeData;
  uint64_t numNodes;
  uint64_t numEdges;
  size_t length;
  bool v2;
  uint64_t numSeeks1, numSeeksDst, numSeeksData;

  Galois::Substrate::SimpleLock lock;

  uint64_t outIndexs(uint64_t node) {
    std::lock_guard<decltype(lock)> lg(lock);
    std::streamoff pos = (4 + node)*sizeof(uint64_t);
    if (file1.tellg() != pos){
       numSeeks1++;
      file1.seekg(pos, file1.beg);
    }
    uint64_t retval;
    file1.read(reinterpret_cast<char*>(&retval), sizeof(uint64_t));
    return retval;
  }

  uint64_t outEdges(uint64_t edge) {
    std::lock_guard<decltype(lock)> lg(lock);
    std::streamoff pos = (4 + numNodes) * sizeof(uint64_t) + edge * (v2 ? sizeof(uint64_t) : sizeof(uint32_t));
    if (fileIndex.tellg() != pos){
       numSeeksDst++;
       fileIndex.seekg(pos, file1.beg);
    }
    if (v2) {
      uint64_t retval;
      fileIndex.read(reinterpret_cast<char*>(&retval), sizeof(uint64_t));
      return retval;
    } else {
      uint32_t retval;
      fileIndex.read(reinterpret_cast<char*>(&retval), sizeof(uint32_t));
      return retval;
    }
  }

  template<typename T>
  T edgeData(uint64_t edge) {
    std::lock_guard<decltype(lock)> lg(lock);
    std::streamoff pos = (4 + numNodes) * sizeof(uint64_t) + numEdges * (v2 ? sizeof(uint64_t) : sizeof(uint32_t));
    //align
    pos = (pos + 7) & ~7;
    pos += edge * sizeof(T);
    if (fileEdgeData.tellg() != pos){
       numSeeksData++;
       fileEdgeData.seekg(pos, file1.beg);
    }
    T retval;
    fileEdgeData.read(reinterpret_cast<char*>(&retval), sizeof(T));
    /*fprintf(stderr, "READ:: %ld[", edge);
    for(int i=0; i<sizeof(T); ++i){
       fprintf(stderr, "%c", reinterpret_cast<char*>(&retval)[i]);
    }
    fprintf(stderr, "]");*/
    return retval;
  }

public:
  typedef boost::counting_iterator<uint32_t> iterator;
  typedef boost::counting_iterator<uint64_t> edge_iterator;
  typedef uint32_t GraphNode;

  OfflineGraph(const std::string& name)
    :file1(name), fileEdgeDst(name), fileEdgeData(name),fileIndex(name),numSeeks1(0), numSeeksDst(0), numSeeksData(0)
  {
    if (!file1.is_open() || !file1.good()) throw "Bad filename";
    uint64_t ver = 0;
    file1.read(reinterpret_cast<char*>(&ver), sizeof(uint64_t));
    file1.seekg(sizeof(uint64_t), file1.cur);
    file1.read(reinterpret_cast<char*>(&numNodes), sizeof(uint64_t));
    file1.read(reinterpret_cast<char*>(&numEdges), sizeof(uint64_t));
    if (ver == 0 || ver > 2) throw "Bad Version";
    v2 = ver == 2;
    if (!file1) throw "Out of data";
    //File length
    file1.seekg(0, file1.end);
    length = file1.tellg();
    if (length < sizeof(uint64_t)*(4+numNodes) + (v2 ? sizeof(uint64_t) : sizeof(uint32_t))*numEdges)
      throw "File too small";
    
  }
  uint64_t num_seeks(){
     std::cout << "Seeks :: " << numSeeks1 << " , " << numSeeksData << " , " << numSeeksDst << " \n";
     return numSeeks1+numSeeksData+numSeeksDst;
  }
  void reset_seek_counters(){
     numSeeks1=numSeeksData=numSeeksDst=0;
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

  template<typename T>
  T getEdgeData(edge_iterator ni) {
    return edgeData<T>(*ni);
  }

};


#endif//_GALOIS_DIST_OFFLINE_GRAPH_
