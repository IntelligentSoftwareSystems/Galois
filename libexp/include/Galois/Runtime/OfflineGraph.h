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
 * @author Loc Hoang <l_hoang@utexas.edu> (prefix range at the bottom)
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

  Galois::Substrate::SimpleLock lock;

  uint64_t outIndexs(uint64_t node) {
    std::lock_guard<decltype(lock)> lg(lock);
    std::streamoff pos = (4 + node)*sizeof(uint64_t);
    if (locEdgeDst != pos){
      numSeeksEdgeDst++;
      fileEdgeDst.seekg(pos, fileEdgeDst.beg);
      locEdgeDst = pos;
    }
    uint64_t retval;
    try {
      fileEdgeDst.read(reinterpret_cast<char*>(&retval), sizeof(uint64_t));
    }
    catch (std::ifstream::failure e) {
      std::cerr << "Exception while reading edge destinations:" << e.what() << "\n";
      std::cerr << "IO error flags: EOF " << fileEdgeDst.eof() << " FAIL " << fileEdgeDst.fail() << " BAD " << fileEdgeDst.bad() << "\n";
    }
    auto numBytesRead = fileEdgeDst.gcount();
    assert(numBytesRead == sizeof(uint64_t));
    locEdgeDst += numBytesRead;
    numBytesReadEdgeDst += numBytesRead;
    return retval;
  }

  uint64_t outEdges(uint64_t edge) {
    std::lock_guard<decltype(lock)> lg(lock);
    std::streamoff pos = (4 + numNodes) * sizeof(uint64_t) + edge * (v2 ? sizeof(uint64_t) : sizeof(uint32_t));
    if (locIndex != pos){
       numSeeksIndex++;
       fileIndex.seekg(pos, fileEdgeDst.beg);
       locIndex = pos;
    }
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
    std::streamoff pos = (4 + numNodes) * sizeof(uint64_t) + numEdges * (v2 ? sizeof(uint64_t) : sizeof(uint32_t));
    //align
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
    }
    catch (std::ifstream::failure e) {
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

  OfflineGraph(const std::string& name)
    :fileEdgeDst(name, std::ios_base::binary), fileIndex(name, std::ios_base::binary), fileEdgeData(name, std::ios_base::binary),
     locEdgeDst(0), locIndex(0), locEdgeData(0),
     numSeeksEdgeDst(0), numSeeksIndex(0), numSeeksEdgeData(0),
     numBytesReadEdgeDst(0), numBytesReadIndex(0), numBytesReadEdgeData(0)

  {
    if (!fileEdgeDst.is_open() || !fileEdgeDst.good()) throw "Bad filename";
    if (!fileIndex.is_open() || !fileIndex.good()) throw "Bad filename";
    if (!fileEdgeData.is_open() || !fileEdgeData.good()) throw "Bad filename";
    fileEdgeDst.exceptions(std::ifstream::eofbit | std::ifstream::failbit | std::ifstream::badbit);
    fileIndex.exceptions(std::ifstream::eofbit | std::ifstream::failbit | std::ifstream::badbit);
    fileEdgeData.exceptions(std::ifstream::eofbit | std::ifstream::failbit | std::ifstream::badbit);

    uint64_t ver = 0;

    try {
      fileEdgeDst.read(reinterpret_cast<char*>(&ver), sizeof(uint64_t));
      fileEdgeDst.read(reinterpret_cast<char*>(&sizeEdgeData), sizeof(uint64_t));
      fileEdgeDst.read(reinterpret_cast<char*>(&numNodes), sizeof(uint64_t));
      fileEdgeDst.read(reinterpret_cast<char*>(&numEdges), sizeof(uint64_t));
    }
    catch (std::ifstream::failure e) {
      std::cerr << "Exception while reading graph header:" << e.what() << "\n";
      std::cerr << "IO error flags: EOF " << fileEdgeDst.eof() << " FAIL " << fileEdgeDst.fail() << " BAD " << fileEdgeDst.bad() << "\n";
    }

    if (ver == 0 || ver > 2) throw "Bad Version";
    v2 = ver == 2;
    if (!fileEdgeDst) throw "Out of data";
    //File length
    fileEdgeDst.seekg(0, fileEdgeDst.end);
    length = fileEdgeDst.tellg();
    if (length < sizeof(uint64_t)*(4+numNodes) + (v2 ? sizeof(uint64_t) : sizeof(uint32_t))*numEdges)
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
  void reset_seek_counters(){
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

};


class OfflineGraphWriter {
  std::fstream file;
  uint64_t numNodes, numEdges;
  bool smallData;

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


public:
OfflineGraphWriter(const std::string& name, bool use32=false) :file(name, std::ios_base::in | std::ios_base::out | std::ios_base::binary | std::ios_base::trunc), numNodes(0), numEdges(0), smallData(use32) {
    if (!file.is_open() || !file.good()) throw "Bad filename";
    uint64_t ver = 1;
    uint64_t etSize = smallData ? sizeof(float) : sizeof(double);
    file.write(reinterpret_cast<char*>(&ver), sizeof(uint64_t));
    file.write(reinterpret_cast<char*>(&etSize), sizeof(uint64_t));
    file.write(reinterpret_cast<char*>(&numNodes), sizeof(uint64_t));
    file.write(reinterpret_cast<char*>(&numEdges), sizeof(uint64_t));
    file.seekg(0, std::ios_base::beg);
  }

  ~OfflineGraphWriter() {}

  //sets the number of nodes and edges.  points to an container of edge counts
  void setCounts(std::deque<uint64_t> edgeCounts) {
    edgeOffsets = std::move(edgeCounts);
    numNodes = edgeOffsets.size();
    numEdges = std::accumulate(edgeOffsets.begin(),edgeOffsets.end(),0);
    std::partial_sum(edgeOffsets.begin(), edgeOffsets.end(), edgeOffsets.begin());
    file.seekg(sizeof(uint64_t)*2, std::ios_base::beg);
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
};


} // namespace Graph


// In namespace Galois
/**
 * Given an offline graph, find a division of nodes based on edges.
 *
 * @param graph OfflineGraph representation
 * @param begin Beginning of iterator to graph
 * @param division_id The division that you want the range for
 * @param num_divisions The total number of divisions you are working with
 * @returns A pair of 2 iterators that correspond to the beginning and the
 * end of the range for the division_id (end not inclusive)
 */
template<typename IterTy>
std::pair<IterTy, IterTy> prefix_range(Galois::Graph::OfflineGraph& graph,
                                       IterTy begin,
                                       uint32_t division_id, 
                                       uint32_t num_divisions) {
  uint32_t total_nodes = graph.size();
  assert(division_id < num_divisions);

  // Single division case
  if (num_divisions == 1) {
    //printf("For division %u/%u we have begin %u and end %lu with %lu edges\n", 
    //       division_id, num_divisions - 1, 
    //       0, total_nodes, edge_prefix_sum.back());
    return std::make_pair(begin, graph.end());
  }

  // Case where we have more divisions than nodes
  if (num_divisions > total_nodes) {
    // assign one element per division, i.e. division id n gets assigned to
    // element n (if element n exists, else range is nothing)
    if (division_id < total_nodes) {
      IterTy node_to_get = begin + division_id;
      //// this division gets a element
      //if (division_id == 0) {
      //  printf("For division %u/%u we have begin %u and end %u with %lu edges\n", 
      //         division_id, num_divisions - 1, division_id, division_id + 1,
      //         edge_prefix_sum[0]);
      //} else {
      //  printf("For division %u/%u we have begin %u and end %u with %lu edges\n", 
      //         division_id, num_divisions - 1, division_id, division_id + 1,
      //         edge_prefix_sum[division_id] - edge_prefix_sum[division_id - 1]);

      //}
      return std::make_pair(node_to_get, node_to_get + 1);
    } else {
      // this division gets no element
      //printf("For division %u/%u we have begin %lu and end %lu with 0 edges\n", 
      //       division_id, num_divisions - 1, total_nodes, total_nodes);
      return std::make_pair(graph.end(), graph.end());
    }
  }

  // To determine range for some element n, you have to determine
  // range for elements 1 through n-1...
  uint32_t current_division = 0;
  uint64_t begin_element = 0;

  uint64_t accounted_edges = 0;
  uint64_t current_element = 0;

  // theoretically how many edges we want to distributed to each division
  uint64_t edges_per_division = graph.sizeEdges() / num_divisions;

  //printf("Optimally want %lu edges per division\n", edges_per_division);
  // TODO also could keep track of begin's prefix sum for debug printing, but
  // would require more seeks
  uint64_t current_prefix_sum = -1;
  uint64_t last_prefix_sum = -1;
  uint64_t current_processed = 0;

  while (current_element < total_nodes && current_division < num_divisions) {
    uint64_t elements_remaining = total_nodes - current_element;
    uint32_t divisions_remaining = num_divisions - current_division;

    assert(elements_remaining >= divisions_remaining);

    if (divisions_remaining == 1) {
      // assign remaining elements to last division
      assert(current_division == num_divisions - 1); 
      printf("For division %u/%u we have begin %lu and end as the end "
             "getting the rest of the graph",
              division_id, num_divisions - 1, begin_element);
      //if (current_element != 0) {
      //  printf("For division %u/%u we have begin %lu and end %lu with "
      //         "%lu edges\n", 
      //         division_id, num_divisions - 1, begin_element, 
      //         current_element + 1, 
      //         edge_prefix_sum[current_element] - 
      //           edge_prefix_sum[begin_element - 1]);
      //} else {
      //  printf("For division %u/%u we have begin %lu and end %lu with "
      //         "%lu edges\n", 
      //         division_id, num_divisions - 1, begin_element, 
      //         current_element + 1, edge_prefix_sum.back());
      //}

      return std::make_pair(begin + current_element, graph.end());
    } else if ((total_nodes - current_element) == divisions_remaining) {
      // Out of elements to assign: finish up assignments (at this point,
      // each remaining division gets 1 element except for the current
      // division which may have some already)

      for (uint32_t i = 0; i < divisions_remaining; i++) {
        if (current_division == division_id) {
          // TODO get these prints working, but would require more disk seeks
          //if (begin_element != 0) {
          //  printf("For division %u/%u we have begin %lu and end %lu with "
          //         "%lu edges\n", 
          //         division_id, num_divisions - 1, begin_element, 
          //         current_element + 1, 
          //         edge_prefix_sum[current_element] - 
          //           edge_prefix_sum[begin_element - 1]);
          //} else {
          //  printf("For division %u/%u we have begin %lu and end %lu with "
          //         "%lu edges\n", 
          //         division_id, num_divisions - 1, begin_element, 
          //         current_element + 1, edge_prefix_sum[current_element]);
          //}

          return std::make_pair(begin + begin_element, 
                                begin + current_element + 1);
        } else {
          current_division++;
          begin_element = current_element + 1;
          current_element++;
        }
      }

      // shouldn't get out here...
      assert(false);
    }

    // Determine various edge count numbers
    uint64_t element_edges;

    if (current_element > 0) {
      // update last_prefix_sum once only
      if (current_processed != current_element) {
        last_prefix_sum = current_prefix_sum;
        current_prefix_sum = *(graph.edge_end(current_element));
        current_processed = current_element;
      }

      element_edges = current_prefix_sum - last_prefix_sum;
      //element_edges = edge_prefix_sum[current_element] - 
      //                edge_prefix_sum[current_element - 1];
    } else {
      current_prefix_sum = *(graph.edge_end(0));
      element_edges = current_prefix_sum;
    }

    uint64_t edge_count_without_current;
    if (current_element > 0) {
      edge_count_without_current = current_prefix_sum - accounted_edges - 
                                   element_edges;
      //edge_count_without_current = edge_prefix_sum[current_element] -
      //                             accounted_edges - element_edges;
    } else {
      edge_count_without_current = 0;
    }
 
    // if this element has a lot of edges, determine if it should go to
    // this division or the next (don't want to cause this division to get
    // too much)
    if (element_edges > (3 * edges_per_division / 4)) {
      // if this current division + edges of this element is too much,
      // then do not add to this division but rather the next one
      if (edge_count_without_current > (edges_per_division / 2)) {

        // finish up this division; its last element is the one before this
        // one
        if (current_division == division_id) {
          printf("For division %u/%u we have begin %lu and end %lu with "
                 "%lu edges\n", division_id, num_divisions - 1, begin_element, 
                 current_element, edge_count_without_current);
  
          return std::make_pair(begin + begin_element, 
                                begin + current_element);
        } else {
          assert(current_division < division_id);

          // this is safe (i.e. won't access -1) as you should never enter this 
          // conditional if current element is still 0
          accounted_edges = last_prefix_sum;
          //accounted_edges = edge_prefix_sum[current_element - 1];
          begin_element = current_element;
          current_division++;

          continue;
        }
      }
    }

    // handle this element by adding edges to running sums
    uint64_t edge_count_with_current = edge_count_without_current + 
                                       element_edges;

    if (edge_count_with_current >= edges_per_division) {
      // this division has enough edges after including the current
      // node; finish up

      if (current_division == division_id) {
        printf("For division %u/%u we have begin %lu and end %lu with %lu edges\n", 
               division_id, num_divisions - 1, begin_element, 
               current_element + 1, edge_count_with_current);

        return std::make_pair(begin + begin_element, 
                              begin + current_element + 1);
      } else {
        //accounted_edges = edge_prefix_sum[current_element];
        accounted_edges = current_prefix_sum;
        // beginning element of next division
        begin_element = current_element + 1;
        current_division++;
      }
    }

    current_element++;
  }

  // You shouldn't get out here.... (something should be returned before
  // this....)
  assert(false);














































  // TODO
}

} // namespace Galois

#endif//_GALOIS_DIST_OFFLINE_GRAPH_
