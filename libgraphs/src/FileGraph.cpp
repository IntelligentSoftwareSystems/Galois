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
 * @author Loc Hoang <l_hoang@utexas.edu>
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

/**
 * Default file graph constructor which initializes fields to null values.
 */
FileGraph::FileGraph()
  : sizeofEdge(0), numNodes(0), numEdges(0),
    outIdx(0), outs(0), edgeData(0), graphVersion(-1),
    nodeOffset(0), edgeOffset(0)
{ }

/**
 * Construct graph from another FileGraph
 *
 * @param o Other filegraph to initialize from.
 */
FileGraph::FileGraph(const FileGraph& o) {
  fromArrays(o.outIdx, o.numNodes, o.outs, o.numEdges, o.edgeData, 
             o.sizeofEdge, o.nodeOffset, o.edgeOffset, true, o.graphVersion);
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
    outIdx(0), outs(0), edgeData(0), graphVersion(-1),
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

/**
 * Given an mmap'd version of the graph, read it in from memory.
 *
 * TODO better documentation?
 */
void FileGraph::fromMem(void* m, uint64_t node_offset, uint64_t edge_offset, 
                        uint64_t lenlimit) {
  uint64_t* fptr = (uint64_t*)m;
  graphVersion = convert_le64toh(*fptr++);

  if (graphVersion != 1 && graphVersion != 2) {
    GALOIS_DIE("unknown file version ", graphVersion);
  }

  sizeofEdge = convert_le64toh(*fptr++);
  numNodes = convert_le64toh(*fptr++);
  numEdges = convert_le64toh(*fptr++);
  nodeOffset = node_offset;
  edgeOffset = edge_offset;
  outIdx = fptr;

  // move over to outgoing edge data and save it
  fptr += numNodes;
  outs = (void*)fptr;

  // skip memory differently depending on file version
  if (graphVersion == 1) {
    uint32_t* fptr32 = (uint32_t*)fptr;
    fptr32 += numEdges + numEdges % 2;
    if (!lenlimit || lenlimit > numEdges + ((char*)fptr32 - (char*)m))
      edgeData = (char*)fptr32;
    else
      edgeData = 0;
  } else {
    uint64_t* fptr64 = (uint64_t*)fptr;
    fptr64 += numEdges + numEdges % 2;

    if (!lenlimit || lenlimit > numEdges + ((char*)fptr64 - (char*)m))
      edgeData = (char*)fptr64;
    else
      edgeData = 0;
  }
}

static size_t rawBlockSize(size_t numNodes, size_t numEdges, size_t sizeofEdgeData,
                           int graphVersion) {
  // header size: version, sizeof_edge_data, numNodes, numEdges, all uint64_t
  size_t bytes = sizeof(uint64_t) * 4; 

  // node data
  bytes += sizeof(uint64_t) * numNodes;

  if (graphVersion == 1) {
    bytes += sizeof(uint32_t) * numEdges;

    if (numEdges % 2)
      bytes += sizeof(uint32_t); // padding
  } else if (graphVersion == 2) {
    bytes += sizeof(uint64_t) * numEdges;
    // no padding necessary in version 2 TODO verify this
  } else {
    GALOIS_DIE("graph version not set", graphVersion);
  }

  bytes += sizeofEdgeData * numEdges;
  return bytes;
}

void* FileGraph::fromGraph(FileGraph& g, size_t sizeof_edge_data) {
  return fromArrays(g.outIdx, g.numNodes, g.outs, g.numEdges, g.edgeData, 
                    sizeof_edge_data, g.nodeOffset, g.edgeOffset, true, 
                    g.graphVersion);
}

void* FileGraph::fromArrays(uint64_t* out_idx, uint64_t num_nodes, void* outs, 
                            uint64_t num_edges, char* edge_data, 
                            size_t sizeof_edge_data, uint64_t node_offset, 
                            uint64_t edge_offset, bool converted, 
                            int oGraphVersion) {
  size_t bytes = rawBlockSize(num_nodes, num_edges, sizeof_edge_data, oGraphVersion);

  char* base = (char*)mmap_big(nullptr, bytes, PROT_READ | PROT_WRITE, 
                               _MAP_ANON | MAP_PRIVATE, -1, 0);
  if (base == MAP_FAILED)
    GALOIS_SYS_DIE("failed allocating graph");

  mappings.push_back({base, bytes});
  
  uint64_t* fptr = (uint64_t*) base;
  // set header info
  if (oGraphVersion == 1) {
    *fptr++ = convert_htole64(1);
  } else if (oGraphVersion == 2) {
    *fptr++ = convert_htole64(2);
  } else {
    GALOIS_DIE("unknown file version to fromArrays", oGraphVersion);
  }
  *fptr++ = convert_htole64(sizeof_edge_data);
  *fptr++ = convert_htole64(num_nodes);
  *fptr++ = convert_htole64(num_edges);

  // copy node data
  if (converted) {
    memcpy(fptr, out_idx, sizeof(*out_idx) * num_nodes);
    fptr += num_nodes;
  } else {
    for (size_t i = 0; i < num_nodes; ++i)
      *fptr++ = convert_htole64(out_idx[i]);
  }

  // TODO verify
  char* fptr0;

  // copy edge destinations
  if (oGraphVersion == 1) {
    uint32_t* fptr32 = (uint32_t*) fptr;

    if (converted) {
      //memcpy(fptr32, outs, sizeof(*outs) * num_edges);
      memcpy(fptr32, outs, sizeof(uint32_t) * num_edges);
      fptr32 += num_edges;
    } else {
      for (size_t i = 0; i < num_edges; ++i)
        *fptr32++ = convert_htole32(((uint32_t*)outs)[i]);
    }

    // padding
    if (num_edges % 2)
      fptr32 += 1;

    fptr0 = (char*)fptr32;
  } else {
    // should be version 2; otherwise would have died above
    // note fptr is already typed as uint64_t*...
    if (converted) {
      memcpy(fptr, outs, sizeof(uint64_t) * num_edges);
      fptr += num_edges;
    } else {
      for (size_t i = 0; i < num_edges; ++i)
        *fptr++ = convert_htole64(((uint64_t*)outs)[i]);
    }

    // padding
    if (num_edges % 2)
      fptr += 1;

    fptr0 = (char*)fptr;
  }

  // copy edge data if necessary
  if (edge_data)
    memcpy(fptr0, edge_data, sizeof_edge_data * num_edges);

  // "load" filegraph from our constructed base pointer
  fromMem(base, node_offset, edge_offset, 0);
  // graph version should be set in from mem

  assert(graphVersion == oGraphVersion);

  return edgeData;
}

/**
 * Given a file name, mmap the graph into memory.
 *
 * @param filename Graph file to load
 */
void FileGraph::fromFile(const std::string& filename) {
  int fd = open(filename.c_str(), O_RDONLY);
  if (fd == -1)
    GALOIS_SYS_DIE("failed opening ", "'", filename, "'");
  fds.push_back(fd);

  struct stat buf;
  if (fstat(fd, &buf) == -1)
    GALOIS_SYS_DIE("failed reading ", "'", filename, "'");

  // mmap file, then load from mem using fromMem function
  void* base = mmap_big(nullptr, buf.st_size, PROT_READ, 
                        MAP_PRIVATE | MAP_POPULATE, fd, 0);
  if (base == MAP_FAILED)
    GALOIS_SYS_DIE("failed reading ", "'", filename, "'");
  mappings.push_back({base, static_cast<size_t>(buf.st_size)});

  fromMem(base, 0, 0, buf.st_size);
}

/**
 * Load graph data from a given offset
 *
 * @param fd File descriptor to load
 * @param offset Offset into file to load
 * @param length Amount of the file to laod
 * @param mappings Mappings structure that tracks the things we have mmap'd
 * @returns Pointer to mmap'd location in memory
 */
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

/**
 * Loads/mmaps particular portions of a graph corresponding to a node
 * range and edge range into memory.
 *
 * @param filename File to load
 * @param nrange Node range to load
 * @param erange Edge range to load
 */
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

  // at this point we should have access to graphVersion...

  // Adjust metadata to correspond to part
  uint64_t partNumNodes = std::distance(nrange.first, nrange.second);
  uint64_t partNumEdges = std::distance(erange.first, erange.second);
  size_t length = partNumNodes * sizeof(uint64_t);
  offset_t offset = headerSize + nodeOffset * sizeof(uint64_t);
  outIdx = static_cast<uint64_t*>(loadFromOffset(fd, offset, length, mappings));

  // TODO verify correctness
  if (graphVersion == 1) {
    length = partNumEdges * sizeof(uint32_t);
    offset = headerSize + numNodes * sizeof(uint64_t) + edgeOffset * sizeof(uint32_t);
    outs = loadFromOffset(fd, offset, length, mappings);
  } else if (graphVersion == 2) {
    length = partNumEdges * sizeof(uint64_t);
    offset = headerSize + numNodes * sizeof(uint64_t) + edgeOffset * sizeof(uint64_t);
    outs = loadFromOffset(fd, offset, length, mappings);
  } else {
    GALOIS_DIE("unknown file version at partFromFile", graphVersion);
  }

  edgeData = 0;
  if (sizeofEdge) {
    length = partNumEdges * sizeofEdge;
    offset = rawBlockSize(numNodes, numEdges, 0, graphVersion) + sizeofEdge * edgeOffset;
    edgeData = static_cast<char*>(loadFromOffset(fd, offset, length, mappings));
  }

  numNodes = partNumNodes;
  numEdges = partNumEdges;
}

// Note this is the original find index; kept for divideByEdge
size_t FileGraph::findIndex(size_t nodeSize, size_t edgeSize, size_t targetSize, 
                            size_t lb, size_t ub) {
  while (lb < ub) {
    size_t mid = lb + (ub - lb) / 2;
    size_t num_edges = *edge_begin(mid) - edgeOffset;
    size_t size = num_edges * edgeSize + (mid - nodeOffset) * nodeSize;
    if (size < targetSize)
      lb = mid + 1;
    else
      ub = mid;
  }
  return lb;
}

auto 
FileGraph::divideByNode(size_t nodeSize, size_t edgeSize, size_t id, size_t total)
-> GraphRange {
  std::vector<unsigned> dummy;
  // note this calls into another findIndex (not the one directly above)....
  return Galois::Graph::divideNodesBinarySearch<FileGraph, uint64_t>(
    *this, nodeSize, edgeSize, id, total, dummy, nodeOffset, edgeOffset);

  //size_t size = numNodes * nodeSize + numEdges * edgeSize;
  //size_t block = (size + total - 1) / total;
  //size_t aa = numEdges;
  //size_t ea = numEdges;
  //size_t bb = findIndex(nodeSize, edgeSize, block * id, 0, numNodes);
  //size_t eb = findIndex(nodeSize, edgeSize, block * (id + 1), bb, numNodes);
  //if (bb != eb) {
  //  aa = *edge_begin(bb);
  //  ea = *edge_end(eb-1);
  //}
  //if (false) {
  //  Substrate::gInfo("(", id, "/", total, ") ", bb, " ", eb, " ", eb - bb);
  //}
  //return GraphRange(NodeRange(iterator(bb), iterator(eb)), EdgeRange(edge_iterator(aa), edge_iterator(ea)));
}

/**
 * Divides nodes only considering edges.
 *
 * Note that it may potentially not return all nodes in the graph (it will 
 * return up to the last node with edges).
 */
auto FileGraph::divideByEdge(size_t nodeSize, size_t edgeSize, size_t id, 
                             size_t total) -> std::pair<NodeRange, EdgeRange> {
  size_t size = numEdges;
  size_t block = (size + total - 1) / total;
  size_t aa = block*id;
  size_t ea = std::min(block * (id + 1), static_cast<size_t>(numEdges));
  size_t bb = findIndex(0, 1, aa, 0, numNodes);
  size_t eb = findIndex(0, 1, ea, bb, numNodes);

  if (true) {
    Substrate::gInfo("(", id, "/", total, ") [", bb, " ", eb, " ", eb - bb, 
                     "], [", aa, " ", ea, " ", ea - aa, "]");
  }

  return GraphRange(NodeRange(iterator(bb), iterator(eb)), 
                    EdgeRange(edge_iterator(aa), edge_iterator(ea)));
}

/**
 * Write current contents to a file
 *
 * @param file File to write to
 */
// FIXME: perform host -> le on data
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

// TODO verify
uint64_t FileGraph::getEdgeIdx(GraphNode src, GraphNode dst) {
  if (graphVersion == 1) {
    for (auto ii = (uint32_t*)raw_neighbor_begin(src), 
              ei = (uint32_t*)raw_neighbor_end(src); 
         ii != ei; 
         ++ii) {
      if (convert_le32toh(*ii) == dst)
        return std::distance((uint32_t*)outs, ii);
    }

    return ~static_cast<uint64_t>(0);
  } else if (graphVersion == 2) {
    for (auto ii = (uint64_t*)raw_neighbor_begin(src), 
              ei = (uint64_t*)raw_neighbor_end(src); 
         ii != ei; 
         ++ii) {
      if (convert_le64toh(*ii) == dst)
        return std::distance((uint64_t*)outs, ii);

    }

    return ~static_cast<uint64_t>(0);
  } else {
    GALOIS_DIE("unknown file version at getEdgeIdx", graphVersion);
  }
}

/**
 * Touch pages of a buffer to page them in.
 *
 * @param buf Buffer to touch
 * @param len Length to touch
 * @param stride How much to stride when touching pages
 */
static void pageInReadOnly(void* buf, size_t len, size_t stride) {
  volatile char* ptr = reinterpret_cast<volatile char*>(buf);
  for (size_t i = 0; i < len; i += stride)
    ptr[i];
}

void FileGraph::pageInByNode(size_t id, size_t total, size_t sizeofEdgeData) {
  size_t edgeSize = 0;
  if (graphVersion == 1) {
    edgeSize = sizeof(uint32_t);
  } else if (graphVersion == 2) {
    edgeSize = sizeof(uint64_t);
  } else {
    GALOIS_DIE("unknown file version at pageInByNode", graphVersion);
  }

  auto r = divideByNode(sizeof(uint64_t), sizeofEdgeData + edgeSize, id, 
                        total).first;
  
  size_t ebegin = *edge_begin(*r.first);
  size_t eend = ebegin;
  if (r.first != r.second)
    eend = *edge_end(*r.second - 1);

  pageInReadOnly(outIdx + *r.first, 
                 std::distance(r.first, r.second) * sizeof(*outIdx), 
                 Runtime::pagePoolSize());

  // TODO verify
  if (graphVersion == 1) {
    pageInReadOnly((uint32_t*)outs + ebegin, 
                   (eend - ebegin) * sizeof(uint32_t), 
                   Runtime::pagePoolSize());
  } else {
    pageInReadOnly((uint64_t*)outs + ebegin, 
                   (eend - ebegin) * sizeof(uint64_t), 
                   Runtime::pagePoolSize());
  }

  pageInReadOnly(edgeData + ebegin * sizeofEdgeData, 
                 (eend - ebegin) * sizeofEdgeData, 
                 Runtime::pagePoolSize());
}

// TODO verify
void* FileGraph::raw_neighbor_begin(GraphNode N) {
  if (graphVersion == 1) {
    return &(((uint32_t*)outs)[*edge_begin(N)]);
  } else if (graphVersion == 2) {
    return &(((uint64_t*)outs)[*edge_begin(N)]);
  } else {
    GALOIS_DIE("unknown file version at raw_neighbor_begin", graphVersion);
  }

  return nullptr;
}

// TODO verify
void* FileGraph::raw_neighbor_end(GraphNode N) {
  if (graphVersion == 1) {
    return &(((uint32_t*)outs)[*edge_end(N)]);
  } else if (graphVersion == 2) {
    return &(((uint64_t*)outs)[*edge_end(N)]);
  } else {
    GALOIS_DIE("unknown file version at raw_neighbor_end", graphVersion);
  }

  return nullptr;
}

FileGraph::edge_iterator FileGraph::edge_begin(GraphNode N) {
  size_t idx = 0;
  if (N > nodeOffset) {
    numBytesReadIndex += 8;
    idx = std::min(convert_le64toh(outIdx[N-1-nodeOffset]), 
                   static_cast<uint64_t>(edgeOffset + numEdges)) - 
          edgeOffset;
  } else if (N != nodeOffset) {
    printf("WARNING: reading node out of bounds for this file graph");
    // TODO die here?
  }
  return edge_iterator(idx);
}

FileGraph::edge_iterator FileGraph::edge_end(GraphNode N) {
  size_t idx = 0;
  if (N >= nodeOffset) {
    numBytesReadIndex += 8;
    idx = std::min(convert_le64toh(outIdx[N-nodeOffset]), 
                   static_cast<uint64_t>(edgeOffset + numEdges)) - 
          edgeOffset;
  } else {
    printf("WARNING: reading node out of bounds for this file graph");
    // TODO die here?
  }
  return edge_iterator(idx);
}

// TODO verify
FileGraph::GraphNode FileGraph::getEdgeDst(edge_iterator it) {
  if (graphVersion == 1) {
    numBytesReadEdgeDst += 4;
    // can safely return 32 bit as 64 bit
    return convert_le32toh(((uint32_t*)outs)[*it]);
  } else if (graphVersion == 2) {
    numBytesReadEdgeDst += 8;
    return convert_le64toh(((uint64_t*)outs)[*it]);
  } else {
    GALOIS_DIE("unknown file version at getEdgeDst", graphVersion);
  }

  return -1;
}

// TODO CURRENTLY ONLY VERSION 1 SUPPORT
FileGraph::node_id_iterator FileGraph::node_id_begin() const {
  return boost::make_transform_iterator(&((uint32_t*)outs)[0], 
                                        Convert32());
}

// TODO CURRENTLY ONLY VERSION 1 SUPPORT
FileGraph::node_id_iterator FileGraph::node_id_end() const {
  return boost::make_transform_iterator(&((uint32_t*)outs)[numEdges], 
                                        Convert32());
}

FileGraph::edge_id_iterator FileGraph::edge_id_begin() const {
  return boost::make_transform_iterator(&outIdx[0], Convert64());
}

FileGraph::edge_id_iterator FileGraph::edge_id_end() const {
  return boost::make_transform_iterator(&outIdx[numNodes], Convert64());
}

bool FileGraph::hasNeighbor(GraphNode N1, GraphNode N2) {
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
