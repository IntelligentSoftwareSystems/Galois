/** File graph -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2013, The University of Texas at Austin. All rights reserved.
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
 * Graph serialized to a file.
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */
#include "Galois/Graph/FileGraph.h"
#include "Galois/Runtime/mm/Mem.h"

#include <cassert>

#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <unistd.h>

using namespace Galois::Graph;

//File format V1:
//version (1) {uint64_t LE}
//EdgeType size {uint64_t LE}
//numNodes {uint64_t LE}
//numEdges {uint64_t LE}
//outindexs[numNodes] {uint64_t LE} (outindex[nodeid] is index of first edge for nodeid + 1 (end interator.  node 0 has an implicit start iterator of 0.
//outedges[numEdges] {uint32_t LE}
//potential padding (32bit max) to Re-Align to 64bits
//EdgeType[numEdges] {EdgeType size}

FileGraph::FileGraph()
  : masterMapping(0), masterLength(0), masterFD(0),
    outIdx(0), outs(0), edgeData(0),
    numEdges(0), numNodes(0)
{
}

FileGraph::~FileGraph() {
  if (masterMapping)
    munmap(masterMapping, masterLength);
  if (masterFD)
    close(masterFD);
}

//FIXME: perform le -> host on data here too
void FileGraph::parse(void* m) {
  //parse file
  uint64_t* fptr = (uint64_t*)m;
  __attribute__((unused)) uint64_t version = le64toh(*fptr); fptr++;
  assert(version == 1);
  sizeofEdge = le64toh(*fptr); fptr++;
  numNodes = le64toh(*fptr); fptr++;
  numEdges = le64toh(*fptr); fptr++;
  outIdx = fptr;
  fptr += numNodes;
  uint32_t* fptr32 = (uint32_t*)fptr;
  outs = fptr32; 
  fptr32 += numEdges;
  if (numEdges % 2)
    fptr32 += 1;
  edgeData = (char*)fptr32;
}

void FileGraph::structureFromMem(void* mem, size_t len, bool clone) {
  masterLength = len;

  if (clone) {
    int _MAP_BASE = MAP_ANONYMOUS | MAP_PRIVATE;
#ifdef MAP_POPULATE
    _MAP_BASE |= MAP_POPULATE;
#endif
    
    void* m = mmap(0, masterLength, PROT_READ | PROT_WRITE, _MAP_BASE, -1, 0);
    if (m == MAP_FAILED) {
      GALOIS_SYS_DIE("failed copying graph");
    }
    memcpy(m, mem, len);
    parse(m);
    masterMapping = m;
  } else {
    parse(mem);
    masterMapping = mem;
  }
}

void* FileGraph::structureFromGraph(FileGraph& g, size_t sizeof_edge_data) {
  // Allocate
  size_t common = g.masterLength - (g.sizeofEdge * g.numEdges);
  size_t len = common + (sizeof_edge_data * g.numEdges);
  int _MAP_BASE = MAP_ANONYMOUS | MAP_PRIVATE;
#ifdef MAP_POPULATE
  _MAP_BASE |= MAP_POPULATE;
#endif
  void* m = mmap(0, len, PROT_READ | PROT_WRITE, _MAP_BASE, -1, 0);
  if (m == MAP_FAILED) {
    GALOIS_SYS_DIE("failed copying graph");
  }
  memcpy(m, g.masterMapping, common);
  uint64_t* fptr = (uint64_t*)m;
  fptr[1] = sizeof_edge_data; // Update sizeof(edgeData)
  //parse(m);
  structureFromMem(m, len, false);

  return edgeData;
}

void* FileGraph::structureFromArrays(uint64_t* out_idx, uint64_t num_nodes,
      uint32_t* outs, uint64_t num_edges, size_t sizeof_edge_data) {
  //version
  uint64_t version = 1;
  uint64_t nBytes = sizeof(uint64_t) * 4; // version, sizeof_edge_data, numNodes, numEdges

  nBytes += sizeof(uint64_t) * num_nodes;
  nBytes += sizeof(uint32_t) * num_edges;
  if (num_edges % 2) {
    nBytes += sizeof(uint32_t); // padding
  }
  nBytes += sizeof_edge_data * num_edges;
 
  int _MAP_BASE = MAP_ANONYMOUS | MAP_PRIVATE;
#ifdef MAP_POPULATE
  _MAP_BASE |= MAP_POPULATE;
#endif
  
  char* t = (char*) mmap(0, nBytes, PROT_READ | PROT_WRITE, _MAP_BASE, -1, 0);
  if (t == MAP_FAILED) {
    t = 0;
    GALOIS_SYS_DIE("failed allocating graph");
  }
  char* base = t;
  memcpy(t, &version, sizeof(version));
  t += sizeof(version);
  memcpy(t, &sizeof_edge_data, sizeof(sizeof_edge_data));
  t += sizeof(sizeof_edge_data);
  memcpy(t, &num_nodes, sizeof(num_nodes));
  t += sizeof(num_nodes);
  memcpy(t, &num_edges, sizeof(num_edges));
  t += sizeof(num_edges);
  memcpy(t, out_idx, sizeof(*out_idx) * num_nodes);
  t += sizeof(*out_idx) * num_nodes;
  memcpy(t, outs, sizeof(*outs) * num_edges);
  if (num_edges % 2) {
    t += sizeof(uint32_t); // padding
  }
  
  structureFromMem(base, nBytes, false);
  return edgeData;
}

void FileGraph::structureFromFile(const std::string& filename, bool preFault) {
  masterFD = open(filename.c_str(), O_RDONLY);
  if (masterFD == -1) {
    GALOIS_SYS_DIE("failed opening ", filename);
  }

  struct stat buf;
  int f = fstat(masterFD, &buf);
  if (f == -1) {
    GALOIS_SYS_DIE("failed reading ", filename);
  }
  masterLength = buf.st_size;

  int _MAP_BASE = MAP_PRIVATE;
#ifdef MAP_POPULATE
  if (preFault)
    _MAP_BASE |= MAP_POPULATE;
#endif
  
  void* m = mmap(0, masterLength, PROT_READ, _MAP_BASE, masterFD, 0);
  if (m == MAP_FAILED) {
    m = 0;
    GALOIS_SYS_DIE("failed reading ", filename);
  }
  parse(m);
  masterMapping = m;

#ifndef MAP_POPULATE
  if (preFault) {
    Runtime::MM::pageIn(m, masterLength);
  }
#endif
}

size_t FileGraph::findIndex(size_t nodeSize, size_t edgeSize, size_t targetSize, size_t lb, size_t ub) {
  while (lb < ub) {
    size_t mid = lb + (ub - lb) / 2;
    size_t num_edges = *edge_end(mid);
    size_t size = num_edges * edgeSize + (mid+1) * nodeSize;
    if (size < targetSize)
      lb = mid + 1;
    else
      ub = mid;
  }
  return lb;
}

auto 
FileGraph::divideBy(size_t nodeSize, size_t edgeSize, unsigned id, unsigned total) -> std::pair<iterator,iterator>
{
  size_t size = numNodes * nodeSize + numEdges * edgeSize;
  size_t block = (size + total - 1) / total;
  size_t bb = findIndex(nodeSize, edgeSize, block * id, 0, numNodes);
  size_t eb;
  if (id + 1 == total)
    eb = numNodes;
  else
    eb = findIndex(nodeSize, edgeSize, block * (id + 1), bb, numNodes);
  //Runtime::LL::gInfo("(", id, "/", total, ") ", bb, " ", eb, " ", numNodes);
  return std::make_pair(iterator(bb), iterator(eb));
}

//FIXME: perform host -> le on data
void FileGraph::structureToFile(const std::string& file) {
  ssize_t retval;
  //ASSUME LE machine
  mode_t mode = S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH;
  int fd = open(file.c_str(), O_WRONLY | O_CREAT | O_TRUNC, mode);
  size_t total = masterLength;
  char* ptr = (char*) masterMapping;
  while (total) {
    retval = write(fd, ptr, total);
    if (retval == -1) {
      GALOIS_SYS_DIE("failed writing to ", file);
    } else if (retval == 0) {
      GALOIS_DIE("ran out of space writing to ", file);
    }
    total -= retval;
    ptr += retval;
  }
  close(fd);
}

void FileGraph::swap(FileGraph& other) {
  std::swap(masterMapping, other.masterMapping);
  std::swap(masterLength, other.masterLength);
  std::swap(sizeofEdge, other.sizeofEdge);
  std::swap(masterFD, other.masterFD);
  std::swap(outIdx, other.outIdx);
  std::swap(outs, other.outs);
  std::swap(edgeData, other.edgeData);
  std::swap(numEdges, other.numEdges);
  std::swap(numNodes, other.numNodes);
}

void FileGraph::cloneFrom(FileGraph& other) {
  structureFromMem(other.masterMapping, other.masterLength, true);
}

uint64_t FileGraph::getEdgeIdx(GraphNode src, GraphNode dst) const {
  for (uint32_t* ii = raw_neighbor_begin(src),
	 *ee = raw_neighbor_end(src); ii != ee; ++ii)
    if (le32toh(*ii) == dst)
      return std::distance(outs, ii);
  return ~static_cast<uint64_t>(0);
}

uint32_t* FileGraph::raw_neighbor_begin(GraphNode N) const {
  return (N == 0) ? &outs[0] : &outs[le64toh(outIdx[N-1])];
}

uint32_t* FileGraph::raw_neighbor_end(GraphNode N) const {
  return &outs[le64toh(outIdx[N])];
}

FileGraph::edge_iterator FileGraph::edge_begin(GraphNode N) const {
  return edge_iterator(N == 0 ? 0 : le64toh(outIdx[N-1]));
}

FileGraph::edge_iterator FileGraph::edge_end(GraphNode N) const {
  return edge_iterator(le64toh(outIdx[N]));
}

FileGraph::GraphNode FileGraph::getEdgeDst(edge_iterator it) const {
  return le32toh(outs[*it]);
}

FileGraph::node_id_iterator FileGraph::node_id_begin() const {
  return boost::make_transform_iterator(&outs[0], Convert32());
}

FileGraph::node_id_iterator FileGraph::node_id_end() const {
  return boost::make_transform_iterator(&outs[numEdges], Convert32());
}

FileGraph::edge_id_iterator FileGraph::edge_id_begin() const {
  return boost::make_transform_iterator(&outIdx[0], Convert64());
}

FileGraph::edge_id_iterator FileGraph::edge_id_end() const {
  return boost::make_transform_iterator(&outIdx[numNodes], Convert64());
}

bool FileGraph::hasNeighbor(GraphNode N1, GraphNode N2) const {
  return getEdgeIdx(N1,N2) != ~static_cast<uint64_t>(0);
}

FileGraph::iterator FileGraph::begin() const {
  return iterator(0);
}

FileGraph::iterator FileGraph::end() const {
  return iterator(numNodes);
}
