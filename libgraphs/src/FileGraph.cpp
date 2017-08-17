/** File graph -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galois is a framework to exploit
 * amorphous data-parallelism in irregular programs.
 *
 * Galois is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, version 2.1 of the
 * License.
 *
 * Galois is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with Galois.  If not, see
 * <http://www.gnu.org/licenses/>.
 *
 * @section Copyright
 *
 * Copyright (C) 2015, The University of Texas at Austin. All rights
 * reserved.
 *
 * @section Description
 *
 * Graph serialized to a file.
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */

#include "Galois/Graphs/FileGraph.h"
#include "Galois/Substrate/PageAlloc.h"

#include <cassert>
#include <fstream>

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
//version (1 or 2) {uint64_t LE}
//EdgeType size {uint64_t LE}
//numNodes {uint64_t LE}
//numEdges {uint64_t LE}
//outindexs[numNodes] {uint64_t LE} (outindex[nodeid] is index of first edge for nodeid + 1 (end interator.  node 0 has an implicit start iterator of 0.
//outedges[numEdges] {uint32_t LE or uint64_t LE for ver == 2}
//potential padding (32bit max) to Re-Align to 64bits
//EdgeType[numEdges] {EdgeType size}

FileGraph::FileGraph()
  : sizeofEdge(0), numNodes(0), numEdges(0),
    outIdx(0), outs(0), edgeData(0),
    nodeOffset(0), edgeOffset(0)
{
}

FileGraph::FileGraph(const FileGraph& o) {
  fromArrays(o.outIdx, o.numNodes, o.outs, o.numEdges, o.edgeData, o.sizeofEdge, o.nodeOffset, o.edgeOffset, true);
}

FileGraph& FileGraph::operator=(const FileGraph& other) {
  if (this != &other) {
    FileGraph tmp(other);
    *this = std::move(tmp);
  }
  return *this;
}

FileGraph::FileGraph(FileGraph&& other)
  : sizeofEdge(0), numNodes(0), numEdges(0),
    outIdx(0), outs(0), edgeData(0),
    nodeOffset(0), edgeOffset(0)
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
  std::swap(nodeOffset, o.nodeOffset);
  std::swap(edgeOffset, o.edgeOffset);
}

void FileGraph::fromMem(void* m, uint32_t node_offset, uint64_t edge_offset, uint64_t lenlimit) {
  uint64_t* fptr = (uint64_t*)m;
  uint64_t version = convert_le64toh(*fptr++);
  if (version != 1)
    GALOIS_DIE("unknown file version ", version);
  sizeofEdge = convert_le64toh(*fptr++);
  numNodes = convert_le64toh(*fptr++);
  numEdges = convert_le64toh(*fptr++);
  nodeOffset = node_offset;
  edgeOffset = edge_offset;
  outIdx = fptr;
  fptr += numNodes;
  uint32_t* fptr32 = (uint32_t*)fptr;
  outs = fptr32;
  fptr32 += numEdges + numEdges % 2;
  if (!lenlimit || lenlimit > numEdges + ((char*)fptr32 - (char*)m))
    edgeData = (char*)fptr32;
  else
    edgeData = 0;
}

static size_t rawBlockSize(size_t numNodes, size_t numEdges, size_t sizeofEdgeData) {
  size_t bytes = sizeof(uint64_t) * 4; // version, sizeof_edge_data, numNodes, numEdges
  bytes += sizeof(uint64_t) * numNodes;
  bytes += sizeof(uint32_t) * numEdges;
  if (numEdges % 2)
    bytes += sizeof(uint32_t); // padding
  bytes += sizeofEdgeData * numEdges;
  return bytes;
}

void* FileGraph::fromGraph(FileGraph& g, size_t sizeof_edge_data) {
  return fromArrays(g.outIdx, g.numNodes, g.outs, g.numEdges, g.edgeData, sizeof_edge_data, g.nodeOffset, g.edgeOffset, true);
}

void* FileGraph::fromArrays(
    uint64_t* out_idx, uint64_t num_nodes,
    uint32_t* outs, uint64_t num_edges,
    char* edge_data, size_t sizeof_edge_data, 
    uint32_t node_offset, uint64_t edge_offset,    
    bool converted) 
{
  size_t bytes = rawBlockSize(num_nodes, num_edges, sizeof_edge_data);
  char* base = (char*) mmap_big(nullptr, bytes, PROT_READ | PROT_WRITE, _MAP_ANON | MAP_PRIVATE, -1, 0);
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

  fromMem(base, node_offset, edge_offset, 0);
  return edgeData;
}

void FileGraph::fromFile(const std::string& filename) {
  int fd = open(filename.c_str(), O_RDONLY);
  if (fd == -1)
    GALOIS_SYS_DIE("failed opening ", "'", filename, "'");
  fds.push_back(fd);

  struct stat buf;
  if (fstat(fd, &buf) == -1)
    GALOIS_SYS_DIE("failed reading ", "'", filename, "'");

  void* base = mmap_big(nullptr, buf.st_size, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);
  if (base == MAP_FAILED)
    GALOIS_SYS_DIE("failed reading ", "'", filename, "'");
  mappings.push_back({base, static_cast<size_t>(buf.st_size)});

  fromMem(base, 0, 0, buf.st_size);
}

template<typename Mappings>
static void* loadFromOffset(int fd, offset_t offset, size_t length, Mappings& mappings) {
  // mmap needs page-aligned offsets
  offset_t aligned = offset & ~static_cast<offset_t>(Galois::Substrate::allocSize() - 1);
  offset_t alignment = offset - aligned;
  length += alignment;
  void *base = mmap_big(nullptr, length, PROT_READ, MAP_PRIVATE, fd, aligned);
  if (base == MAP_FAILED)
    GALOIS_SYS_DIE("failed allocating for fd ", fd);
  mappings.push_back({base, length});
  return static_cast<char*>(base) + alignment;
}

void FileGraph::partFromFile(const std::string& filename, NodeRange nrange, 
                             EdgeRange erange) {
  int fd = open(filename.c_str(), O_RDONLY);
  if (fd == -1)
    GALOIS_SYS_DIE("failed opening ", "'", filename, "'");
  fds.push_back(fd);

  size_t headerSize = 4 * sizeof(uint64_t);
  void* base = mmap(nullptr, headerSize, PROT_READ, MAP_PRIVATE, fd, 0);
  if (base == MAP_FAILED)
    GALOIS_SYS_DIE("failed reading ", "'", filename, "'");
  mappings.push_back({base, headerSize});

  // Read metadata of whole graph
  fromMem(base, *nrange.first, *erange.first, 0);

  // Adjust metadata to correspond to part
  uint64_t partNumNodes = std::distance(nrange.first, nrange.second);
  uint64_t partNumEdges = std::distance(erange.first, erange.second);
  size_t length = partNumNodes * sizeof(uint64_t);
  offset_t offset = headerSize + nodeOffset * sizeof(uint64_t);
  outIdx = static_cast<uint64_t*>(loadFromOffset(fd, offset, length, mappings));

  length = partNumEdges * sizeof(uint32_t);
  offset = headerSize + numNodes * sizeof(uint64_t) + edgeOffset * sizeof(uint32_t);
  outs = static_cast<uint32_t*>(loadFromOffset(fd, offset, length, mappings));

  edgeData = 0;
  if (sizeofEdge) {
    length = partNumEdges * sizeofEdge;
    offset = rawBlockSize(numNodes, numEdges, 0) + sizeofEdge * edgeOffset;
    edgeData = static_cast<char*>(loadFromOffset(fd, offset, length, mappings));
  }

  numNodes = partNumNodes;
  numEdges = partNumEdges;
}

size_t FileGraph::findIndex(size_t nodeSize, size_t edgeSize, size_t targetSize, size_t lb, size_t ub) {
  while (lb < ub) {
    size_t mid = lb + (ub - lb) / 2;
    size_t num_edges = *edge_begin(mid) - edgeOffset;
    size_t size = num_edges * edgeSize + (mid-nodeOffset) * nodeSize;
    if (size < targetSize)
      lb = mid + 1;
    else
      ub = mid;
  }
  return lb;
}

auto 
FileGraph::divideByNode(size_t nodeSize, size_t edgeSize, size_t id, size_t total)
-> GraphRange
{
  size_t size = numNodes * nodeSize + numEdges * edgeSize;
  size_t block = (size + total - 1) / total;
  size_t aa = numEdges;
  size_t ea = numEdges;
  size_t bb = findIndex(nodeSize, edgeSize, block * id, 0, numNodes);
  size_t eb = findIndex(nodeSize, edgeSize, block * (id + 1), bb, numNodes);
  if (bb != eb) {
    aa = *edge_begin(bb);
    ea = *edge_end(eb-1);
  }
  if (false) {
    Substrate::gInfo("(", id, "/", total, ") ", bb, " ", eb, " ", eb - bb);
  }
  return GraphRange(NodeRange(iterator(bb), iterator(eb)), EdgeRange(edge_iterator(aa), edge_iterator(ea)));
}

auto
FileGraph::divideByEdge(size_t nodeSize, size_t edgeSize, size_t id, size_t total)
-> std::pair<NodeRange, EdgeRange>
{
  size_t size = numEdges;
  size_t block = (size + total - 1) / total;
  size_t aa = block*id;
  size_t ea = std::min(block * (id + 1), static_cast<size_t>(numEdges));
  size_t bb = findIndex(0, 1, aa, 0, numNodes);
  size_t eb = findIndex(0, 1, ea, bb, numNodes);

  if (true) {
    Substrate::gInfo("(", id, "/", total, ") [", bb, " ", eb, " ", eb - bb, "], [", aa, " ", ea, " ", ea - aa, "]");
  }
  return GraphRange(NodeRange(iterator(bb), iterator(eb)), EdgeRange(edge_iterator(aa), edge_iterator(ea)));
}

//FIXME: perform host -> le on data
void FileGraph::toFile(const std::string& file) {
  // FIXME handle files with multiple mappings
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
  for (auto ii = raw_neighbor_begin(src), ei = raw_neighbor_end(src); ii != ei; ++ii) {
    if (convert_le32toh(*ii) == dst)
      return std::distance(outs, ii);
  }
  return ~static_cast<uint64_t>(0);
}

static void pageInReadOnly(void* buf, size_t len, size_t stride) {
  volatile char* ptr = reinterpret_cast<volatile char*>(buf);
  for (size_t i = 0; i < len; i += stride)
    ptr[i];
}

void FileGraph::pageInByNode(size_t id, size_t total, size_t sizeofEdgeData) {
  auto r = divideByNode(
    sizeof(uint64_t),
    sizeofEdgeData + sizeof(uint32_t),
    id, total).first;
  
  size_t ebegin = *edge_begin(*r.first);
  size_t eend = ebegin;
  if (r.first != r.second)
    eend = *edge_end(*r.second - 1);

  pageInReadOnly(outIdx + *r.first, std::distance(r.first, r.second) * sizeof(*outIdx), Runtime::pagePoolSize());
  pageInReadOnly(outs + ebegin, (eend - ebegin) * sizeof(*outs), Runtime::pagePoolSize());
  pageInReadOnly(edgeData + ebegin * sizeofEdgeData, (eend - ebegin) * sizeofEdgeData, Runtime::pagePoolSize());
}

uint32_t* FileGraph::raw_neighbor_begin(GraphNode N) const {
  return &outs[*edge_begin(N)];
}

uint32_t* FileGraph::raw_neighbor_end(GraphNode N) const {
  return &outs[*edge_end(N)];
}

FileGraph::edge_iterator FileGraph::edge_begin(GraphNode N) const {
  size_t idx = 0;
  if (N > nodeOffset)
    idx = std::min(convert_le64toh(outIdx[N-1-nodeOffset]), 
                   static_cast<uint64_t>(edgeOffset + numEdges)) - 
          edgeOffset;
  return edge_iterator(idx);
}

FileGraph::edge_iterator FileGraph::edge_end(GraphNode N) const {
  size_t idx = 0;
  if (N >= nodeOffset)
    idx = std::min(convert_le64toh(outIdx[N-nodeOffset]), 
                   static_cast<uint64_t>(edgeOffset + numEdges)) - 
          edgeOffset;
  return edge_iterator(idx);
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
  return iterator(nodeOffset);
}

FileGraph::iterator FileGraph::end() const {
  return iterator(nodeOffset + numNodes);
}

} // end Graph
} // end Galois
