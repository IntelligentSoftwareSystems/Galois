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

#ifdef __linux__
#include <linux/mman.h>
#endif
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <unistd.h>

#if defined(MAP_ANONYMOUS)
static const int _MAP_ANON = MAP_ANONYMOUS;
#elif defined(MAP_ANON)
static const int _MAP_ANON = MAP_ANON;
#else
// fail
#endif
#ifdef MAP_POPULATE
static const int _MAP_POP  = MAP_POPULATE;
#else
static const int _MAP_POP  = 0;
#endif
static const int _MAP_BASE = _MAP_ANON | _MAP_POP | MAP_PRIVATE;

#ifdef HAVE_MMAP64
template<typename... Args>
void* mmap_big(Args... args) {
  return mmap64(std::forward<Args>(args)...);
}
typedef off64_t offset_t;
#else
template<typename... Args>
void* mmap_big(Args... args) {
  return mmap(std::forward<Args>(args)...);
}
typedef off_t offset_t;
#endif

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

FileGraph::FileGraph()
  : sizeofEdge(0), numNodes(0), numEdges(0),
    outIdx(0), outs(0), edgeData(0)
{
}

FileGraph::FileGraph(const FileGraph& o) {
  fromArrays(o.outIdx, o.numNodes, o.outs, o.numEdges, o.edgeData, o.sizeofEdge, true);
}

FileGraph& FileGraph::operator=(const FileGraph& other) {
  if (this != &other) {
    FileGraph tmp(*this);
    *this = std::move(tmp);
  }
  return *this;
}

FileGraph::FileGraph(FileGraph&& other)
  : sizeofEdge(0), numNodes(0), numEdges(0),
    outIdx(0), outs(0), edgeData(0)
{
  move_assign(std::move(other));
}

FileGraph& FileGraph::operator=(FileGraph&& other) {
  move_assign(std::move(other));
  return *this;
}

FileGraph::~FileGraph() {
  for (auto& m : mappings)
    munmap(m.ptr, m.len);
  for (auto& fd : fds)
    close(fd);
}

void FileGraph::move_assign(FileGraph&& o) {
  std::swap(mappings, o.mappings);
  std::swap(fds, o.fds);
  std::swap(sizeofEdge, o.sizeofEdge);
  std::swap(numNodes, o.numNodes);
  std::swap(numEdges, o.numEdges);
  std::swap(outIdx, o.outIdx);
  std::swap(outs, o.outs);
  std::swap(edgeData, o.edgeData);
}

void FileGraph::fromMem(void* m) {
  //parse file
  uint64_t* fptr = (uint64_t*)m;
  uint64_t version = convert_le64toh(*fptr++);
  if (version != 1)
    GALOIS_DIE("unknown file version ", version);
  sizeofEdge = convert_le64toh(*fptr++);
  numNodes = convert_le64toh(*fptr++);
  numEdges = convert_le64toh(*fptr++);
  outIdx = fptr;
  fptr += numNodes;
  uint32_t* fptr32 = (uint32_t*)fptr;
  outs = fptr32; 
  fptr32 += numEdges;
  if (numEdges % 2)
    fptr32 += 1;
  edgeData = (char*)fptr32;
}

static inline size_t rawBlockSize(size_t numNodes, size_t numEdges, size_t sizeofEdgeData) {
  size_t bytes = sizeof(uint64_t) * 4; // version, sizeof_edge_data, numNodes, numEdges
  bytes += sizeof(uint64_t) * numNodes;
  bytes += sizeof(uint32_t) * numEdges;
  if (numEdges % 2)
    bytes += sizeof(uint32_t); // padding
  bytes += sizeofEdgeData * numEdges;
  return bytes;
}

void* FileGraph::fromGraph(FileGraph& g, size_t sizeof_edge_data) {
  return fromArrays(g.outIdx, g.numNodes, g.outs, g.numEdges, g.edgeData, sizeof_edge_data, true);
}

void* FileGraph::fromArrays(uint64_t* out_idx, uint64_t num_nodes,
      uint32_t* outs, uint64_t num_edges, char* edge_data, size_t sizeof_edge_data, bool converted) {
  size_t bytes = rawBlockSize(num_nodes, num_edges, sizeof_edge_data);
  char* base = (char*) mmap_big(nullptr, bytes, PROT_READ | PROT_WRITE, _MAP_BASE, -1, 0);
  if (base == MAP_FAILED)
    GALOIS_SYS_DIE("failed allocating graph");
  mappings.push_back({base, bytes});
  
  uint64_t* fptr = (uint64_t*) base;
  *fptr++ = convert_htole64(1);
  *fptr++ = convert_htole64(sizeof_edge_data);
  *fptr++ = convert_htole64(num_nodes);
  *fptr++ = convert_htole64(num_edges);

  if (converted) {
    memcpy(fptr, out_idx, sizeof(*out_idx) * num_nodes);
    fptr += num_nodes;
  } else {
    for (size_t i = 0; i < num_nodes; ++i)
      *fptr++ = convert_htole64(out_idx[i]);
  }

  uint32_t* fptr32 = (uint32_t*) fptr;
  if (converted) {
    memcpy(fptr32, outs, sizeof(*outs) * num_edges);
    fptr32 += num_edges;
  } else {
    for (size_t i = 0; i < num_edges; ++i)
      *fptr32++ = convert_htole32(outs[i]);
  }

  if (num_edges % 2)
    fptr32 += 1;

  char* fptr0 = (char*) fptr32;
  if (edge_data)
    memcpy(fptr0, edge_data, sizeof_edge_data * num_edges);

  fromMem(base);
  return edgeData;
}

void FileGraph::fromFile(const std::string& filename, bool preFault) {
  int fd = open(filename.c_str(), O_RDONLY);
  if (fd == -1)
    GALOIS_SYS_DIE("failed opening ", "'", filename, "'");
  fds.push_back(fd);

  struct stat buf;
  if (fstat(fd, &buf) == -1)
    GALOIS_SYS_DIE("failed reading ", "'", filename, "'");

  void* base = mmap_big(nullptr, buf.st_size, PROT_READ, preFault ? (MAP_PRIVATE | _MAP_POP) : MAP_PRIVATE, fd, 0);
  if (base == MAP_FAILED)
    GALOIS_SYS_DIE("failed reading ", "'", filename, "'");
  mappings.push_back({base, static_cast<size_t>(buf.st_size)});

  fromMem(base);

#ifndef MAP_POPULATE
  if (preFault) {
    Runtime::MM::pageInReadOnly(base, buf.st_size, Galois::Runtime::MM::pageSize);
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
  if (false) {
    Runtime::LL::gInfo("(", id, "/", total, ") ", bb, " ", eb, " ", eb - bb);
  }
  return std::make_pair(iterator(bb), iterator(eb));
}

//FIXME: perform host -> le on data
void FileGraph::toFile(const std::string& file) {
  // XXX handle files with multiple mappings
  GALOIS_ASSERT(mappings.size() == 1);

  ssize_t retval;
  mode_t mode = S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH;
  int fd = open(file.c_str(), O_WRONLY | O_CREAT | O_TRUNC, mode);
  mapping mm = mappings.back();
  mappings.pop_back();

  size_t total = mm.len;
  char* ptr = (char*) mm.ptr;
  while (total) {
    retval = write(fd, ptr, total);
    if (retval == -1) {
      GALOIS_SYS_DIE("failed writing to ", "'", file, "'");
    } else if (retval == 0) {
      GALOIS_DIE("ran out of space writing to ", "'", file, "'");
    }
    total -= retval;
    ptr += retval;
  }
  close(fd);
}

uint64_t FileGraph::getEdgeIdx(GraphNode src, GraphNode dst) const {
  for (uint32_t* ii = raw_neighbor_begin(src),
	 *ee = raw_neighbor_end(src); ii != ee; ++ii)
    if (convert_le32toh(*ii) == dst)
      return std::distance(outs, ii);
  return ~static_cast<uint64_t>(0);
}

void FileGraph::pageIn(unsigned id, unsigned total, size_t sizeofEdgeData) {
  // XXX: by edge or by node ?
  auto r = divideBy(
    sizeof(uint64_t),
    sizeofEdgeData + sizeof(uint32_t),
    id, total);
  
  size_t ebegin = *edge_begin(*r.first);
  size_t eend = ebegin;
  if (r.first != r.second)
    eend = *edge_end(*r.second - 1);

  Runtime::MM::pageInReadOnly(outIdx + *r.first, std::distance(r.first, r.second) * sizeof(*outIdx), Runtime::MM::pageSize);
  Runtime::MM::pageInReadOnly(outs + ebegin, (eend - ebegin) * sizeof(*outs), Runtime::MM::pageSize);
  Runtime::MM::pageInReadOnly(edgeData + ebegin * sizeofEdgeData, (eend - ebegin) * sizeofEdgeData, Runtime::MM::pageSize);
}

uint32_t* FileGraph::raw_neighbor_begin(GraphNode N) const {
  return (N == 0) ? &outs[0] : &outs[convert_le64toh(outIdx[N-1])];
}

uint32_t* FileGraph::raw_neighbor_end(GraphNode N) const {
  return &outs[convert_le64toh(outIdx[N])];
}

FileGraph::edge_iterator FileGraph::edge_begin(GraphNode N) const {
  return edge_iterator(N == 0 ? 0 : convert_le64toh(outIdx[N-1]));
}

FileGraph::edge_iterator FileGraph::edge_end(GraphNode N) const {
  return edge_iterator(convert_le64toh(outIdx[N]));
}

FileGraph::GraphNode FileGraph::getEdgeDst(edge_iterator it) const {
  return convert_le32toh(outs[*it]);
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

} // end Graph
} // end Galois
