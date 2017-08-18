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
 * @author Loc Hoang <l_hoang@utexas.edu>
 */

#ifndef _GALOIS_DIST_OFFLINE_GRAPH_
#define _GALOIS_DIST_OFFLINE_GRAPH_

#include "Galois/Substrate/SimpleLock.h"
#include "Galois/Graphs/Details.h"

#include <cstdint>
#include <iostream>
#include <fstream>
#include <mutex>
#include <numeric>
#include <sys/mman.h>
#include <fcntl.h>

#include <boost/iterator/counting_iterator.hpp>

namespace Galois {
namespace Graph {

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
  std::ifstream fileEdgeDst, fileIndex, fileEdgeData;
  std::streamoff locEdgeDst, locIndex, locEdgeData;

  uint64_t numNodes;
  uint64_t numEdges;
  uint64_t sizeEdgeData;
  size_t length;
  bool v2;
  uint64_t numSeeksEdgeDst, numSeeksIndex, numSeeksEdgeData;
  uint64_t numBytesReadEdgeDst, numBytesReadIndex, numBytesReadEdgeData;

  void* file_buffer;

  Galois::Substrate::SimpleLock lock;

  uint64_t outIndexs(uint64_t node) {
    std::lock_guard<decltype(lock)> lg(lock);
    std::streamoff pos = (4 + node) * sizeof(uint64_t);

    // move to correct position in file
    if (locEdgeDst != pos) {
      numSeeksEdgeDst++;
      fileEdgeDst.seekg(pos, fileEdgeDst.beg);
      locEdgeDst = pos;
    }

    // read the value
    uint64_t retval;
    try {
      fileEdgeDst.read(reinterpret_cast<char*>(&retval), sizeof(uint64_t));
    } catch (std::ifstream::failure e) {
      std::cerr << "Exception while reading edge destinations:" << e.what() << "\n";
      std::cerr << "IO error flags: EOF " << fileEdgeDst.eof() << " FAIL " << fileEdgeDst.fail() << " BAD " << fileEdgeDst.bad() << "\n";
    }

    // metadata update
    auto numBytesRead = fileEdgeDst.gcount();
    assert(numBytesRead == sizeof(uint64_t));
    locEdgeDst += numBytesRead;
    numBytesReadEdgeDst += numBytesRead;

    return retval;
  }

  uint64_t outEdges(uint64_t edge) {
    std::lock_guard<decltype(lock)> lg(lock);
    std::streamoff pos = (4 + numNodes) * sizeof(uint64_t) + edge * 
                           (v2 ? sizeof(uint64_t) : sizeof(uint32_t));

    // move to correct position
    if (locIndex != pos){
       numSeeksIndex++;
       fileIndex.seekg(pos, fileEdgeDst.beg);
       locIndex = pos;
    }

    // v2 reads 64 bits, v1 reads 32 bits
    if (v2) {
      uint64_t retval;
      try {
        fileIndex.read(reinterpret_cast<char*>(&retval), sizeof(uint64_t));
      }
      catch (std::ifstream::failure e) {
        std::cerr << "Exception while reading index:" << e.what() << "\n";
        std::cerr << "IO error flags: EOF " << fileIndex.eof() << " FAIL " << fileIndex.fail() << " BAD " << fileIndex.bad() << "\n";
      }

      auto numBytesRead = fileIndex.gcount();
      assert(numBytesRead == sizeof(uint64_t));
      locIndex += numBytesRead;
      numBytesReadIndex += numBytesRead;
      return retval;
    } else {
      uint32_t retval;
      try {
        fileIndex.read(reinterpret_cast<char*>(&retval), sizeof(uint32_t));
      }
      catch (std::ifstream::failure e) {
        std::cerr << "Exception while reading index:" << e.what() << "\n";
        std::cerr << "IO error flags: EOF " << fileIndex.eof() << " FAIL " << fileIndex.fail() << " BAD " << fileIndex.bad() << "\n";
      }

      auto numBytesRead = fileIndex.gcount();
      assert(numBytesRead == sizeof(uint32_t));
      locIndex += numBytesRead;
      numBytesReadIndex += numBytesRead;
      return retval;
    }
  }

  template<typename T>
  T edgeData(uint64_t edge) {
    assert(sizeof(T) <= sizeEdgeData);
    std::lock_guard<decltype(lock)> lg(lock);
    std::streamoff pos = (4 + numNodes) * sizeof(uint64_t) + numEdges * 
                         (v2 ? sizeof(uint64_t) : sizeof(uint32_t));

    // align + move to correct position
    pos = (pos + 7) & ~7;
    pos += edge * sizeEdgeData;

    if (locEdgeData != pos){
       numSeeksEdgeData++;
       fileEdgeData.seekg(pos, fileEdgeDst.beg);
       locEdgeData = pos;
    }

    T retval;
    try {
      fileEdgeData.read(reinterpret_cast<char*>(&retval), sizeof(T));
    } catch (std::ifstream::failure e) {
      std::cerr << "Exception while reading edge data:" << e.what() << "\n";
      std::cerr << "IO error flags: EOF " << fileEdgeData.eof() << " FAIL " << fileEdgeData.fail() << " BAD " << fileEdgeData.bad() << "\n";
    }

    auto numBytesRead = fileEdgeData.gcount();
    assert(numBytesRead == sizeof(T));
    locEdgeData += numBytesRead;
    numBytesReadEdgeData += numBytesRead;
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

  OfflineGraph(const std::string& name) :
     fileEdgeDst(name, std::ios_base::binary), 
     fileIndex(name, std::ios_base::binary), 
     fileEdgeData(name, std::ios_base::binary),
     locEdgeDst(0), locIndex(0), locEdgeData(0),
     numSeeksEdgeDst(0), numSeeksIndex(0), numSeeksEdgeData(0),
     numBytesReadEdgeDst(0), numBytesReadIndex(0), numBytesReadEdgeData(0)
  {
    if (!fileEdgeDst.is_open() || !fileEdgeDst.good()) 
      throw "Bad filename";
    if (!fileIndex.is_open() || !fileIndex.good()) 
      throw "Bad filename";
    if (!fileEdgeData.is_open() || !fileEdgeData.good()) 
      throw "Bad filename";

    fileEdgeDst.exceptions(std::ifstream::eofbit | 
                           std::ifstream::failbit | 
                           std::ifstream::badbit);
    fileIndex.exceptions(std::ifstream::eofbit | 
                         std::ifstream::failbit | 
                         std::ifstream::badbit);
    fileEdgeData.exceptions(std::ifstream::eofbit | 
                            std::ifstream::failbit | 
                            std::ifstream::badbit);

    uint64_t ver = 0;

    try {
      fileEdgeDst.read(reinterpret_cast<char*>(&ver), sizeof(uint64_t));
      fileEdgeDst.read(reinterpret_cast<char*>(&sizeEdgeData), sizeof(uint64_t));
      fileEdgeDst.read(reinterpret_cast<char*>(&numNodes), sizeof(uint64_t));
      fileEdgeDst.read(reinterpret_cast<char*>(&numEdges), sizeof(uint64_t));
    } catch (std::ifstream::failure e) {
      std::cerr << "Exception while reading graph header:" << e.what() << "\n";
      std::cerr << "IO error flags: EOF " << fileEdgeDst.eof() << 
                   " FAIL " << fileEdgeDst.fail() << 
                   " BAD " << fileEdgeDst.bad() << "\n";
    }

    if (ver == 0 || ver > 2) 
      throw "Bad Version";

    v2 = ver == 2;

    if (!fileEdgeDst) 
      throw "Out of data";

    // File length
    fileEdgeDst.seekg(0, fileEdgeDst.end);
    length = fileEdgeDst.tellg();
    if (length < sizeof(uint64_t) * (4+numNodes) + 
                       (v2 ? sizeof(uint64_t) : sizeof(uint32_t))*numEdges)
      throw "File too small";
    
    fileEdgeDst.seekg(0, std::ios_base::beg);
    fileEdgeData.seekg(0, std::ios_base::beg);
    fileIndex.seekg(0, std::ios_base::beg);
  }

  uint64_t num_seeks(){
     //std::cout << "Seeks :: " << numSeeksEdgeDst << " , " << numSeeksEdgeData << " , " << numSeeksIndex << " \n";
     return numSeeksEdgeDst+numSeeksEdgeData+numSeeksIndex;
  }

  uint64_t num_bytes_read(){
     //std::cout << "Bytes read :: " << numBytesReadEdgeDst << " , " << numBytesReadEdgeData << " , " << numBytesReadIndex << " \n";
     return numBytesReadEdgeDst+numBytesReadEdgeData+numBytesReadIndex;
  }

  void reset_seek_counters() {
     numSeeksEdgeDst=numSeeksEdgeData=numSeeksIndex=0;
     numBytesReadEdgeDst=numBytesReadEdgeData=numBytesReadIndex=0;
  }

  OfflineGraph(OfflineGraph&&) = default;

  size_t size() const { return numNodes; }
  size_t sizeEdges() const { return numEdges; }
  size_t edgeSize() const { return sizeEdgeData; }

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

  Runtime::iterable<NoDerefIterator<edge_iterator>> edges(GraphNode N) {
    return detail::make_no_deref_range(edge_begin(N), edge_end(N));
  }

  template<typename T>
  T getEdgeData(edge_iterator ni) {
    return edgeData<T>(*ni);
  }

  /**
   * Return a suitable index between an upper bound and a lower bound that
   * attempts to get close to the target size (i.e. find a good chunk that
   * corresponds to some size). 
   *
   * @param nodeWeight weight to give to a node in division
   * @param edgeWeight weight to give to an edge in division
   * @param targetWeight The amount of weight we want from the returned index
   * @param lb lower bound to start search from
   * @param ub upper bound to start search from
   */
  size_t findIndex(size_t nodeWeight, size_t edgeWeight, size_t targetWeight, 
                   size_t lb, size_t ub) {
    while (lb < ub) {
      size_t mid = lb + (ub - lb) / 2;
      size_t num_edges;
  
      if (mid != 0) {
        num_edges = *edge_end(mid - 1);
      } else {
        num_edges = 0;
      }
  
      size_t weight = num_edges * edgeWeight + (mid) * nodeWeight;
  
      if (weight < targetWeight)
        lb = mid + 1;
      else
        ub = mid;
    }
    return lb;
  }
  
  // typedefs used by divide by node below
  typedef std::pair<iterator, iterator> NodeRange;
  typedef std::pair<edge_iterator, edge_iterator> EdgeRange;
  typedef std::pair<NodeRange, EdgeRange> GraphRange;
  
  /** 
   * Returns 2 ranges (one for nodes, one for edges) for a particular division.
   * The ranges specify the nodes/edges that a division is responsible for. The
   * function attempts to split them evenly among threads given some kind of
   * weighting
   *
   * @param nodeWeight weight to give to a node in division
   * @param edgeWeight weight to give to an edge in division
   * @param id Division number you want the ranges for
   * @param total Total number of divisions
   * @param scaleFactor Vector specifying if certain divisions should get more 
   * than other divisions
   */
  auto divideByNode(size_t nodeWeight, size_t edgeWeight, size_t id, size_t total, 
                    std::vector<unsigned> scaleFactor = std::vector<unsigned>())
      -> GraphRange {
    assert(total >= 1);
    assert(id >= 0 && id < total);

    // weight of all data
    uint64_t weight = numNodes * nodeWeight + numEdges * edgeWeight;
  
    // determine number of blocks to divide among total divisions
    uint32_t numBlocks = 0;
    if (scaleFactor.empty()) {
      numBlocks = total;

      // scale factor holds a prefix sum of the scale factor
      for (uint32_t i = 0; i < total; i++) {
        scaleFactor.push_back(i + 1);
      }
    } else {
      assert(scaleFactor.size() == total);
      assert(total >= 1);

      // get total number of blocks we need + save a prefix sum of the scale
      // factor vector
      for (uint32_t i = 0; i < total; i++) {
        numBlocks += scaleFactor[i];
        scaleFactor[i] = numBlocks;
      }
    }

    // weight of a block (one block for each division by default; if scale
    // factor specifies something different, then use that instead)
    uint64_t blockWeight = (weight + numBlocks - 1) / numBlocks;

    // lower and upper blocks that this division should get using the prefix
    // sum of scaleFactor calculated above
    uint32_t blockLower;
    if (id != 0) {
      blockLower = scaleFactor[id - 1];
    } else {
      blockLower = 0;
    }
    uint32_t blockUpper = scaleFactor[id];

    assert(blockLower <= blockUpper);

    // find allocation of nodes for this division
    uint32_t nodesLower = findIndex(nodeWeight, edgeWeight, 
                                    blockWeight * blockLower, 0, numNodes);
    uint32_t nodesUpper = findIndex(nodeWeight, edgeWeight,
                                    blockWeight * blockUpper, nodesLower,
                                    numNodes);

    uint64_t edgesLower = numEdges;
    uint64_t edgesUpper = numEdges;
    // correct number of edges based on nodes allocated to division if
    // necessary
    if (nodesLower != nodesUpper) {
      if (nodesLower != 0) {
        edgesLower = *(edge_end(nodesLower - 1));
      } else {
        edgesLower = 0;
      }
      edgesUpper = *(edge_end(nodesUpper - 1));
    }
  
    return GraphRange(NodeRange(iterator(nodesLower), 
                                iterator(nodesUpper)), 
                      EdgeRange(edge_iterator(edgesLower), 
                                edge_iterator(edgesUpper)));
  }
};

class OfflineGraphWriter {
  std::fstream file;
  uint64_t numNodes, numEdges;
  bool smallData;
  uint64_t ver;
  std::vector<uint64_t> bufferDst;
  uint32_t counter;

  std::deque<uint64_t> edgeOffsets;

  std::streamoff offsetOfDst(uint64_t edge) {
    return sizeof(uint64_t)*(4 + numNodes + edge);
  }
  std::streamoff offsetOfData(uint64_t edge) {
    return sizeof(uint64_t) * (4 + numNodes + numEdges) + 
      (smallData ? sizeof(float) : sizeof(double)) * edge;
  }

  void setEdge32(uint64_t src, uint64_t offset, uint64_t dst, uint32_t val) {
    if (src)
      offset += edgeOffsets[src-1];
    file.seekg(offsetOfDst(offset), std::ios_base::beg);
    file.write(reinterpret_cast<char*>(&dst), sizeof(uint64_t));
    file.seekg(offsetOfData(offset), std::ios_base::beg);
    file.write(reinterpret_cast<char*>(&val), sizeof(uint32_t));
  }

  void setEdge64(uint64_t src, uint64_t offset, uint64_t dst, uint64_t val) {
    if (src)
      offset += edgeOffsets[src-1];
    file.seekg(offsetOfDst(offset), std::ios_base::beg);
    file.write(reinterpret_cast<char*>(&dst), sizeof(uint64_t));
    file.seekg(offsetOfData(offset), std::ios_base::beg);
    file.write(reinterpret_cast<char*>(&val), sizeof(uint64_t));
  }


  void setEdge_sorted(uint64_t dst) {
    if(ver == 1){
      uint32_t dst32 = dst;
      file.write(reinterpret_cast<char*>(&dst32), sizeof(uint32_t));
    }
    else{
      file.write(reinterpret_cast<char*>(&dst), sizeof(uint64_t));
    }
  }

  void setEdge_sortedBuffer(){
    if(ver == 1){
      std::vector<uint32_t> tmp(bufferDst.begin(), bufferDst.end());
      file.write(reinterpret_cast<char*>(&tmp[0]), (sizeof(uint32_t)*tmp.size()));
    }
    file.write(reinterpret_cast<char*>(&bufferDst[0]), (sizeof(uint64_t)*bufferDst.size()));
  }

  //void setEdge64_sorted(uint64_t dst) {
    //file.write(reinterpret_cast<char*>(&dst), sizeof(uint32_t));
  //}


public:
OfflineGraphWriter(const std::string& name, bool use32=false, uint64_t _numNodes=0, uint64_t _numEdges=0) :file(name, std::ios_base::in | std::ios_base::out | std::ios_base::binary | std::ios_base::trunc), numNodes(_numNodes), numEdges(_numEdges), smallData(use32),ver(1) {
    if (!file.is_open() || !file.good()) throw "Bad filename";
    uint64_t etSize = smallData ? sizeof(float) : sizeof(double);
    file.write(reinterpret_cast<char*>(&ver), sizeof(uint64_t));
    file.write(reinterpret_cast<char*>(&etSize), sizeof(uint64_t));
    file.write(reinterpret_cast<char*>(&numNodes), sizeof(uint64_t));
    file.write(reinterpret_cast<char*>(&numEdges), sizeof(uint64_t));
    file.seekg(0, std::ios_base::beg);
    //bufferDst.resize(1024);
    //counter = 0;
  }

  ~OfflineGraphWriter() {}

  //sets the number of nodes and edges.  points to an container of edge counts
  void setCounts(std::deque<uint64_t> edgeCounts) {
    edgeOffsets = std::move(edgeCounts);
    numNodes = edgeOffsets.size();
    numEdges = std::accumulate(edgeOffsets.begin(),edgeOffsets.end(),0);
    std::cout << " NUM EDGES  : " << numEdges << "\n";
    std::partial_sum(edgeOffsets.begin(), edgeOffsets.end(), edgeOffsets.begin());
    //Nodes are greater than 2^32 so need ver = 2. 
    if(numNodes >= 4294967296){
      ver = 2;
    }
    else{
      ver = 1;
    }
    std::cout << " USING VERSION : " << ver << "\n";
    uint64_t etSize = 0; //smallData ? sizeof(float) : sizeof(double);
    file.seekg(0, std::ios_base::beg);
    file.write(reinterpret_cast<char*>(&ver), sizeof(uint64_t));
    file.write(reinterpret_cast<char*>(&etSize), sizeof(uint64_t));
    //file.seekg(sizeof(uint64_t)*2, std::ios_base::beg);
    file.write(reinterpret_cast<char*>(&numNodes), sizeof(uint64_t));
    file.write(reinterpret_cast<char*>(&numEdges), sizeof(uint64_t));
    for (auto i : edgeOffsets)
      file.write(reinterpret_cast<char*>(&i), sizeof(uint64_t));
    file.seekg(0, std::ios_base::beg);
  }

  void setEdge(uint64_t src, uint64_t offset, uint64_t dst, uint64_t val) {
    if(smallData)
      setEdge32(src,offset,dst,val);
    else
      setEdge64(src,offset,dst,val);
  }
  void setEdgeSorted(uint64_t dst) {
#if 0
      bufferDst[counter] = dst;
      ++counter;
      if(counter == 1024){
        setEdge_sortedBuffer();
        counter = 0;
      }
#endif
      setEdge_sorted(dst);
  }

  void seekEdgesDstStart(){
    file.seekg(offsetOfDst(0), std::ios_base::beg);
  }
};


} // namespace Graph

//// In namespace Galois
///**
// * Given an offline graph, find a division of nodes based on edges.
// *
// * @param graph OfflineGraph representation
// * @param begin Beginning of iterator to graph
// * @param division_id The division that you want the range for
// * @param num_divisions The total number of divisions you are working with
// * @returns A pair of 2 iterators that correspond to the beginning and the
// * end of the range for the division_id (end not inclusive)
// */
//template<typename IterTy>
//std::pair<IterTy, IterTy> prefix_range(Galois::Graph::OfflineGraph& graph,
//                                       IterTy begin,
//                                       uint32_t division_id, 
//                                       uint32_t num_divisions) {
//  uint64_t total_nodes = graph.size();
//  assert(division_id < num_divisions);
//
//  // Single division case
//  if (num_divisions == 1) {
//    printf("For division %u/%u we have begin %u and end %lu with all edges\n", 
//           division_id, num_divisions - 1, 0, total_nodes);
//    return std::make_pair(begin, *graph.end());
//  }
//
//  // Case where we have more divisions than nodes
//  if (num_divisions > total_nodes) {
//    // assign one element per division, i.e. division id n gets assigned to
//    // element n (if element n exists, else range is nothing)
//    if (division_id < total_nodes) {
//      IterTy node_to_get = begin + division_id;
//      // this division gets a element
//      if (division_id == 0) {
//        printf("For division %u/%u we have begin %u and end %u\n", 
//               division_id, num_divisions - 1, division_id, division_id + 1);
//      } else {
//        printf("For division %u/%u we have begin %u and end %u\n", 
//               division_id, num_divisions - 1, division_id, division_id + 1);
//      }
//      return std::make_pair(node_to_get, node_to_get + 1);
//    } else {
//      // this division gets no element
//      printf("For division %u/%u we have begin %lu and end %lu with 0 edges\n", 
//             division_id, num_divisions - 1, total_nodes, total_nodes);
//      return std::make_pair(*graph.end(), *graph.end());
//    }
//  }
//
//  // To determine range for some element n, you have to determine
//  // range for elements 1 through n-1...
//  uint32_t current_division = 0;
//  uint64_t begin_element = 0;
//
//  uint64_t accounted_edges = 0;
//  uint64_t current_element = 0;
//
//  // theoretically how many edges we want to distributed to each division
//  uint64_t edges_per_division = graph.sizeEdges() / num_divisions;
//
//  printf("Optimally want %lu edges per division\n", edges_per_division);
//
//  uint64_t current_prefix_sum = -1;
//  uint64_t last_prefix_sum = -1;
//  uint64_t current_processed = 0;
//
//  while (current_element < total_nodes && current_division < num_divisions) {
//    uint64_t elements_remaining = total_nodes - current_element;
//    uint32_t divisions_remaining = num_divisions - current_division;
//
//    assert(elements_remaining >= divisions_remaining);
//
//    if (divisions_remaining == 1) {
//      // assign remaining elements to last division
//      assert(current_division == num_divisions - 1); 
//      IterTy the_end = *graph.end();
//
//      printf("For division %u/%u we have begin %lu and end as the end of the "
//             "graph %lu\n", division_id, num_divisions - 1, begin_element, 
//             total_nodes);
//
//      //if (current_element != 0) {
//      //  printf("For division %u/%u we have begin %lu and end %lu with "
//      //         "%lu edges\n", 
//      //         division_id, num_divisions - 1, begin_element, 
//      //         current_element + 1, 
//      //         edge_prefix_sum[current_element] - 
//      //           edge_prefix_sum[begin_element - 1]);
//      //} else {
//      //  printf("For division %u/%u we have begin %lu and end %lu with "
//      //         "%lu edges\n", 
//      //         division_id, num_divisions - 1, begin_element, 
//      //         current_element + 1, edge_prefix_sum.back());
//      //}
//
//      return std::make_pair(begin + current_element, the_end);
//    } else if ((total_nodes - current_element) == divisions_remaining) {
//      // Out of elements to assign: finish up assignments (at this point,
//      // each remaining division gets 1 element except for the current
//      // division which may have some already)
//
//      for (uint32_t i = 0; i < divisions_remaining; i++) {
//        if (current_division == division_id) {
//          if (begin_element != 0) {
//            printf("For division %u/%u we have begin %lu and end %lu\n" ,
//                   division_id, num_divisions - 1, begin_element, 
//                   current_element + 1);
//          } else {
//            printf("For division %u/%u we have begin %lu and end %lu\n",
//                   division_id, num_divisions - 1, begin_element, 
//                   current_element + 1);
//          }
//
//          // TODO get these prints working, but would require more disk seeks
//          //if (begin_element != 0) {
//          //  printf("For division %u/%u we have begin %lu and end %lu with "
//          //         "%lu edges\n", 
//          //         division_id, num_divisions - 1, begin_element, 
//          //         current_element + 1, 
//          //         edge_prefix_sum[current_element] - 
//          //           edge_prefix_sum[begin_element - 1]);
//          //} else {
//          //  printf("For division %u/%u we have begin %lu and end %lu with "
//          //         "%lu edges\n", 
//          //         division_id, num_divisions - 1, begin_element, 
//          //         current_element + 1, edge_prefix_sum[current_element]);
//          //}
//
//          return std::make_pair(begin + begin_element, 
//                                begin + current_element + 1);
//        } else {
//          current_division++;
//          begin_element = current_element + 1;
//          current_element++;
//        }
//      }
//
//      // shouldn't get out here...
//      assert(false);
//    }
//
//    // Determine various edge count numbers
//    uint64_t element_edges;
//
//    if (current_element > 0) {
//      // update last_prefix_sum once only
//      if (current_processed != current_element) {
//        last_prefix_sum = current_prefix_sum;
//        current_prefix_sum = *(graph.edge_end(current_element));
//        current_processed = current_element;
//      }
//
//      element_edges = current_prefix_sum - last_prefix_sum;
//      //element_edges = edge_prefix_sum[current_element] - 
//      //                edge_prefix_sum[current_element - 1];
//    } else {
//      current_prefix_sum = *(graph.edge_end(0));
//      element_edges = current_prefix_sum;
//    }
//
//    uint64_t edge_count_without_current;
//    if (current_element > 0) {
//      edge_count_without_current = current_prefix_sum - accounted_edges - 
//                                   element_edges;
//      //edge_count_without_current = edge_prefix_sum[current_element] -
//      //                             accounted_edges - element_edges;
//    } else {
//      edge_count_without_current = 0;
//    }
// 
//    // if this element has a lot of edges, determine if it should go to
//    // this division or the next (don't want to cause this division to get
//    // too much)
//    if (element_edges > (3 * edges_per_division / 4)) {
//      // if this current division + edges of this element is too much,
//      // then do not add to this division but rather the next one
//      if (edge_count_without_current > (edges_per_division / 2)) {
//
//        // finish up this division; its last element is the one before this
//        // one
//        if (current_division == division_id) {
//          printf("For division %u/%u we have begin %lu and end %lu with "
//                 "%lu edges\n", division_id, num_divisions - 1, begin_element, 
//                 current_element, edge_count_without_current);
//  
//          return std::make_pair(begin + begin_element, 
//                                begin + current_element);
//        } else {
//          assert(current_division < division_id);
//
//          // this is safe (i.e. won't access -1) as you should never enter this 
//          // conditional if current element is still 0
//          accounted_edges = last_prefix_sum;
//          //accounted_edges = edge_prefix_sum[current_element - 1];
//          begin_element = current_element;
//          current_division++;
//
//          continue;
//        }
//      }
//    }
//
//    // handle this element by adding edges to running sums
//    uint64_t edge_count_with_current = edge_count_without_current + 
//                                       element_edges;
//
//    if (edge_count_with_current >= edges_per_division) {
//      // this division has enough edges after including the current
//      // node; finish up
//
//      if (current_division == division_id) {
//        printf("For division %u/%u we have begin %lu and end %lu with %lu edges\n", 
//               division_id, num_divisions - 1, begin_element, 
//               current_element + 1, edge_count_with_current);
//
//        return std::make_pair(begin + begin_element, 
//                              begin + current_element + 1);
//      } else {
//        //accounted_edges = edge_prefix_sum[current_element];
//        accounted_edges = current_prefix_sum;
//        // beginning element of next division
//        begin_element = current_element + 1;
//        current_division++;
//      }
//    }
//
//    current_element++;
//  }
//
//  // You shouldn't get out here.... (something should be returned before
//  // this....)
//  assert(false);
//}

/**
 * Given an offline graph, find a division of nodes based on edges (and 
 * nodes if alpha is non-zero) for all
 * divisions and save the results in a passed in vector.
 *
 * @param graph OfflineGraph representation
 * @param begin Beginning of graph
 * @param end End of graph
 * @param num_divisions The total number of divisions you are working with
 * @param division_vector Vector to hold distribution of nodes among divisions
 * @param node_alpha Weight of a node relative to an edge in distribition
 * decisions
 * @param scale_factor Optional scale factor that weights division of nodes
 * among partitions
 */
template<typename IterTy>
void prefix_range(Galois::Graph::OfflineGraph& graph, 
                  IterTy begin, IterTy end, uint32_t num_divisions, 
                  std::vector<std::pair<IterTy, IterTy>>& division_vector,
                  uint32_t node_alpha,
                  std::vector<unsigned> scale_factor = std::vector<unsigned>()) {
  uint64_t total_nodes = graph.size();

  // Single division case
  if (num_divisions == 1) {
    printf("all elements to 1 division\n");

    division_vector.push_back(std::make_pair(begin, end));
    assert(division_vector.size() == num_divisions);
    return;
  }

  // Case where we have more divisions than nodes
  if (num_divisions > total_nodes) {
    printf("1 element per division\n");

    // assign one element per division, i.e. division id n gets assigned to
    // element n (if element n exists, else range is nothing)
    for (uint64_t i = 0; i < total_nodes; i++) {
      IterTy node_to_get = begin + i;
      division_vector.push_back(std::make_pair(node_to_get, node_to_get + 1));
    }

    // divisions with no elements
    for (uint64_t i = total_nodes; i < num_divisions; i++) {
      division_vector.push_back(std::make_pair(end, end));
    }

    assert(division_vector.size() == num_divisions);
    return;
  }

  uint32_t num_blocks = 0;

  if (scale_factor.empty()) {
    num_blocks = num_divisions;
    for (uint32_t i = 0; i < num_divisions; i++) {
      scale_factor.push_back(1);
    }
  } else {
    assert(scale_factor.size() == num_divisions);
    assert(num_divisions >= 1);

    for (uint32_t i = 0; i < num_divisions; i++) {
      num_blocks += scale_factor[i];
    }
  }

  // weight of all nodes
  uint64_t node_weight = (uint64_t)total_nodes * (uint64_t)node_alpha;

  uint64_t units_per_block = graph.sizeEdges() / num_blocks + node_weight;
  printf("Want %lu units per block (unit of division)\n", units_per_block);


  uint32_t current_division = 0;
  uint64_t begin_element = 0;

  uint64_t accounted_edges = 0;
  uint32_t accounted_nodes = 0;

  uint64_t current_element = 0;

  uint64_t current_prefix_sum = -1;
  uint64_t last_prefix_sum = -1;
  uint64_t current_processed = 0;

  // theoretically how many edges we want to distributed to each division
  uint64_t units_this_division = scale_factor[current_division] * 
                                 units_per_block;

  //printf("Optimally want %lu units this division\n", units_this_division);

  while (current_element < total_nodes && current_division < num_divisions) {
    uint64_t elements_remaining = total_nodes - current_element;
    uint32_t divisions_remaining = num_divisions - current_division;

    assert(elements_remaining >= divisions_remaining);

    if (divisions_remaining == 1) {
      // assign remaining elements to last division
      printf("For division %u/%u we have begin %lu and end as the end of the "
             "graph %lu\n", current_division, num_divisions - 1, begin_element, 
             total_nodes);

      division_vector.push_back(std::make_pair(begin + current_element, 
                                               end));
      assert(division_vector.size() == num_divisions);
      return;
    } else if ((total_nodes - current_element) == divisions_remaining) {
      // Out of elements to assign: finish up assignments (at this point,
      // each remaining division gets 1 element except for the current
      // division which may have some already)

      for (uint32_t i = 0; i < divisions_remaining; i++) {
        if (begin_element != 0) {
          printf("For division %u/%u we have begin %lu and end %lu\n" ,
                 current_division, num_divisions - 1, begin_element, 
                 current_element + 1);
        } else {
          printf("For division %u/%u we have begin %lu and end %lu\n",
                 current_division, num_divisions - 1, begin_element, 
                 current_element + 1);
        }

        division_vector.push_back(std::make_pair(begin + begin_element, 
                                    begin + current_element + 1));
        current_division++;
        begin_element = current_element + 1;
        current_element++;
      }

      assert(division_vector.size() == num_divisions);
      return; 
    }

    // Determine various edge count numbers
    // num edges this element has
    uint64_t element_edges;

    if (current_element > 0) {
      // update last_prefix_sum once only
      if (current_processed != current_element) {
        last_prefix_sum = current_prefix_sum;
        current_prefix_sum = *(graph.edge_end(current_element));
        current_processed = current_element;
      }

      element_edges = current_prefix_sum - last_prefix_sum;
    } else {
      current_prefix_sum = *(graph.edge_end(0));
      element_edges = current_prefix_sum;
    }

    // num edges this division currenlty has without taking into account 
    // the current element
    uint64_t edge_count_without_current;
    if (current_element > 0) {
      edge_count_without_current = current_prefix_sum - accounted_edges - 
                                   element_edges;
    } else {
      edge_count_without_current = 0;
    }

    // determine current unit count by taking into account nodes already 
    // handled
    uint64_t unit_count_without_current = 
       edge_count_without_current + 
       ((current_element - accounted_nodes) * node_alpha); 

    // include node into weight of this element
    uint64_t element_units = element_edges + node_alpha;

    // if this element has a lot of weight, determine if it should go to
    // this division or the next (don't want to cause this division to get
    // too much)
    if (element_units > (3 * units_this_division / 4)) {
      // if this current division + weight of this element is too much,
      // then do not add to this division but rather the next one
      if (unit_count_without_current > (units_this_division / 2)) {
        // finish up this division; its last element is the one before this
        // one
        printf("For division %u/%u we have begin %lu and end %lu with "
               "%lu edges\n", current_division, num_divisions - 1, begin_element, 
               current_element, edge_count_without_current);
  
        division_vector.push_back(std::make_pair(begin + begin_element, 
                                                 begin + current_element));

        // this is safe (i.e. won't access -1) as you should never enter this 
        // conditional if current element is still 0
        accounted_edges = last_prefix_sum;
        accounted_nodes = current_element;

        begin_element = current_element;
        current_division++;
        units_this_division = scale_factor[current_division] * units_per_block;

        continue;
      }
    }

    // handle this element by adding edges to running sums
    uint64_t edge_count_with_current = edge_count_without_current + 
                                       element_edges;

    // handle this element by adding edges to running sums
    uint64_t unit_count_with_current = unit_count_without_current + 
                                       element_units;

    if (unit_count_with_current >= units_this_division) {
      // this division has enough edges after including the current
      // node; finish up
      printf("For division %u/%u we have begin %lu and end %lu with %lu edges\n", 
             current_division, num_divisions - 1, begin_element, 
             current_element + 1, edge_count_with_current);

      division_vector.push_back(std::make_pair(begin + begin_element, 
                                               begin + current_element + 1));

      accounted_edges = current_prefix_sum;
      accounted_nodes = current_element + 1;

      // beginning element of next division
      begin_element = current_element + 1;
      current_division++;
      units_this_division = scale_factor[current_division] * units_per_block;
    }

    current_element++;
  }

  assert(division_vector.size() == num_divisions);
}

} // namespace Galois

#endif//_GALOIS_DIST_OFFLINE_GRAPH_
