/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
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

/**
 * @file FileGraph.cpp
 *
 * Contains FileGraph.h implementations + other static helper functions
 * for FileGraph.
 */

#include "galois/gIO.h"
#include "galois/graphs/FileGraph.h"
#include "galois/substrate/PageAlloc.h"

#include <cassert>
#include <fstream>

#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <unistd.h>

/**
 * Performs an mmap of all provided arguments.
 */
namespace galois {
namespace graphs {
// Graph file format:
// version (1 or 2) {uint64_t LE}
// EdgeType size {uint64_t LE}
// numNodes {uint64_t LE}
// numEdges {uint64_t LE}
// outindexs[numNodes] {uint64_t LE} (outindex[nodeid] is index of first edge
// for nodeid + 1 (end interator.  node 0 has an implicit start iterator of 0.
// outedges[numEdges] {uint32_t LE or uint64_t LE for ver == 2}
// potential padding (32bit max) to Re-Align to 64bits
// EdgeType[numEdges] {EdgeType size}

FileGraph::FileGraph()
    : sizeofEdge(0), numNodes(0), numEdges(0), outIdx(0), outs(0), edgeData(0),
      graphVersion(-1), nodeOffset(0), edgeOffset(0) {}

FileGraph::FileGraph(const FileGraph& o) {
  fromArrays(o.outIdx, o.numNodes, o.outs, o.numEdges, o.edgeData, o.sizeofEdge,
             o.nodeOffset, o.edgeOffset, true, o.graphVersion);
}

FileGraph& FileGraph::operator=(const FileGraph& other) {
  if (this != &other) {
    FileGraph tmp(other);
    *this = std::move(tmp);
  }
  return *this;
}

FileGraph::FileGraph(FileGraph&& other)
    : sizeofEdge(0), numNodes(0), numEdges(0), outIdx(0), outs(0), edgeData(0),
      graphVersion(-1), nodeOffset(0), edgeOffset(0) {
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
  std::swap(graphVersion, o.graphVersion);
  std::swap(nodeOffset, o.nodeOffset);
  std::swap(edgeOffset, o.edgeOffset);
}

void FileGraph::fromMem(void* m, uint64_t node_offset, uint64_t edge_offset,
                        uint64_t lenlimit) {
  uint64_t* fptr = (uint64_t*)m;
  graphVersion   = convert_le64toh(*fptr++);

  if (graphVersion != 1 && graphVersion != 2) {
    GALOIS_DIE("unknown file version ", graphVersion);
  }

  sizeofEdge = convert_le64toh(*fptr++);
  numNodes   = convert_le64toh(*fptr++);
  numEdges   = convert_le64toh(*fptr++);
  nodeOffset = node_offset;
  edgeOffset = edge_offset;
  outIdx     = fptr;

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

/**
 * Calculate the total size needed for all data.
 *
 * @param numNodes number of nodes to make space for
 * @param numEdges number of edges to make space for
 * @param sizeofEdgeData the size taken by 1 edge for its edge data
 * @param graphVersion the graph version of the file being loaded (determines
 * the size of edge ids)
 *
 * @returns Total size in bytes needed to store graph data
 */
static size_t rawBlockSize(size_t numNodes, size_t numEdges,
                           size_t sizeofEdgeData, int graphVersion) {
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
    GALOIS_DIE("unknown file version: ", graphVersion);
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
  size_t bytes =
      rawBlockSize(num_nodes, num_edges, sizeof_edge_data, oGraphVersion);

  char* base = (char*)mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                           _MAP_ANON | MAP_PRIVATE, -1, 0);
  if (base == MAP_FAILED)
    GALOIS_SYS_DIE("failed allocating graph");

  mappings.push_back({base, bytes});

  uint64_t* fptr = (uint64_t*)base;
  // set header info
  if (oGraphVersion == 1) {
    *fptr++ = convert_htole64(1);
  } else if (oGraphVersion == 2) {
    *fptr++ = convert_htole64(2);
  } else {
    GALOIS_DIE("unknown file version: ", oGraphVersion);
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
    uint32_t* fptr32 = (uint32_t*)fptr;

    if (converted) {
      // memcpy(fptr32, outs, sizeof(*outs) * num_edges);
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

void FileGraph::fromFile(const std::string& filename) {
  int fd = open(filename.c_str(), O_RDONLY);
  if (fd == -1)
    GALOIS_SYS_DIE("failed opening ", "'", filename, "'");
  fds.push_back(fd);

  struct stat buf;
  if (fstat(fd, &buf) == -1)
    GALOIS_SYS_DIE("failed reading ", "'", filename, "'");

  // mmap file, then load from mem using fromMem function
  int _MAP_BASE = MAP_PRIVATE;
#ifdef MAP_POPULATE
  _MAP_BASE |= MAP_POPULATE;
#endif
  void* base = mmap(nullptr, buf.st_size, PROT_READ, _MAP_BASE, fd, 0);
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
template <typename Mappings>
static void* loadFromOffset(int fd, offset_t offset, size_t length,
                            Mappings& mappings) {
  // mmap needs page-aligned offsets
  offset_t aligned =
      offset & ~static_cast<offset_t>(galois::substrate::allocSize() - 1);
  offset_t alignment = offset - aligned;
  length += alignment;
  void* base = mmap(nullptr, length, PROT_READ, MAP_PRIVATE, fd, aligned);
  if (base == MAP_FAILED)
    GALOIS_SYS_DIE("failed allocating for fd ", fd);
  mappings.push_back({base, length});
  return static_cast<char*>(base) + alignment;
}

/**
 * Makes multiple threads page in specific portions of a buffer of memory.
 * Useful for NUMA-aware architectures.
 *
 * @param ptr buffer to page in
 * @param length amount of data to page in
 * @param hugePageSize size of a huge page (what is being paged in)
 * @param numThreads number of threads to use when paging in memory
 */
static void pageInterleaved(void* ptr, uint64_t length, uint32_t hugePageSize,
                            unsigned int numThreads) {
  galois::substrate::getThreadPool().run(
      numThreads, [ptr, length, hugePageSize, numThreads]() {
        auto myID = galois::substrate::ThreadPool::getTID();

        volatile char* cptr = reinterpret_cast<volatile char*>(ptr);

        // round robin page distribution among threads (e.g. thread 0 gets
        // a page, then thread 1, then thread n, then back to thread 0 and
        // so on until the end of the region)
        for (size_t x = hugePageSize * myID; x < length;
             x += hugePageSize * numThreads)
          // this should do an access
          cptr[x];
      });
}

void FileGraph::partFromFile(const std::string& filename, NodeRange nrange,
                             EdgeRange erange, bool numaMap) {
  int fd = open(filename.c_str(), O_RDONLY);
  if (fd == -1)
    GALOIS_SYS_DIE("failed opening ", "'", filename, "'");
  fds.push_back(fd);

  size_t headerSize = 4 * sizeof(uint64_t);
  void* base        = mmap(nullptr, headerSize, PROT_READ, MAP_PRIVATE, fd, 0);
  if (base == MAP_FAILED)
    GALOIS_SYS_DIE("failed reading ", "'", filename, "'");
  mappings.push_back({base, headerSize});

  // Read metadata of whole graph
  fromMem(base, *nrange.first, *erange.first, 0);

  // at this point we should have access to graphVersion...

  // Adjust metadata to correspond to part
  uint64_t partNumNodes = std::distance(nrange.first, nrange.second);
  uint64_t partNumEdges = std::distance(erange.first, erange.second);
  size_t length         = partNumNodes * sizeof(uint64_t);
  offset_t offset       = headerSize + nodeOffset * sizeof(uint64_t);
  outIdx = static_cast<uint64_t*>(loadFromOffset(fd, offset, length, mappings));

  // TODO verify correctness
  if (graphVersion == 1) {
    length = partNumEdges * sizeof(uint32_t);
    offset = headerSize + numNodes * sizeof(uint64_t) +
             edgeOffset * sizeof(uint32_t);
    outs = loadFromOffset(fd, offset, length, mappings);
  } else if (graphVersion == 2) {
    length = partNumEdges * sizeof(uint64_t);
    offset = headerSize + numNodes * sizeof(uint64_t) +
             edgeOffset * sizeof(uint64_t);
    outs = loadFromOffset(fd, offset, length, mappings);
  } else {
    GALOIS_DIE("unknown file version: ", graphVersion);
  }

  edgeData = 0;
  if (sizeofEdge) {
    length = partNumEdges * sizeofEdge;
    offset = rawBlockSize(numNodes, numEdges, 0, graphVersion) +
             sizeofEdge * edgeOffset;
    edgeData = static_cast<char*>(loadFromOffset(fd, offset, length, mappings));
  }

  numNodes = partNumNodes;
  numEdges = partNumEdges;

  // do interleaved numa allocation with current number of threads
  if (numaMap) {
    unsigned int numThreads   = galois::runtime::activeThreads;
    const size_t hugePageSize = 2 * 1024 * 1024; // 2MB

    void* ptr;

    // doesn't really matter if only 1 thread; i.e. do nothing i
    // that case
    if (numThreads != 1) {
      // node pointer to edge dest array
      ptr    = (void*)outIdx;
      length = numNodes * sizeof(uint64_t);

      pageInterleaved(ptr, length, hugePageSize, numThreads);

      // edge dest array
      ptr = (void*)outs;
      if (graphVersion == 1) {
        length = numEdges * sizeof(uint32_t);
      } else {
        // v2
        length = numEdges * sizeof(uint64_t);
      }

      pageInterleaved(ptr, length, hugePageSize, numThreads);

      // edge data (if it exists)
      if (sizeofEdge) {
        ptr    = (void*)edgeData;
        length = numEdges * sizeofEdge;

        pageInterleaved(ptr, length, hugePageSize, numThreads);
      }
    }
  }
}

size_t FileGraph::findIndex(size_t nodeSize, size_t edgeSize, size_t targetSize,
                            size_t lb, size_t ub) {
  while (lb < ub) {
    size_t mid = lb + (ub - lb) / 2;
    // edge begin assumes global id, so add nodeoffset to it as we work with
    // local ids
    size_t num_edges = *edge_begin(mid + nodeOffset);
    size_t size      = (num_edges * edgeSize) + (mid * nodeSize);
    if (size < targetSize)
      lb = mid + 1;
    else
      ub = mid;
  }
  return lb;
}

auto FileGraph::divideByNode(size_t nodeSize, size_t edgeSize, size_t id,
                             size_t total) -> GraphRange {
  std::vector<unsigned> dummy_scale_factor; // dummy passed in to function call

  return galois::graphs::divideNodesBinarySearch(
      numNodes, numEdges, nodeSize, edgeSize, id, total, outIdx,
      dummy_scale_factor, edgeOffset);
}

auto FileGraph::divideByEdge(size_t, size_t, size_t id, size_t total)
    -> std::pair<NodeRange, EdgeRange> {
  size_t size  = numEdges;
  size_t block = (size + total - 1) / total;
  size_t aa    = block * id;
  size_t ea    = std::min(block * (id + 1), static_cast<size_t>(numEdges));

  // note these use local node ids (numNodes is made local by partFromFile if
  // it was called)
  size_t bb = findIndex(0, 1, aa, 0, numNodes);
  size_t eb = findIndex(0, 1, ea, bb, numNodes);

  if (true) {
    galois::gInfo("(", id, "/", total, ") [", bb, " ", eb, " ", eb - bb, "], [",
                  aa, " ", ea, " ", ea - aa, "]");
  }

  return GraphRange(NodeRange(iterator(bb), iterator(eb)),
                    EdgeRange(edge_iterator(aa), edge_iterator(ea)));
}

void FileGraph::toFile(const std::string& file) {
  // FIXME handle files with multiple mappings
  GALOIS_ASSERT(mappings.size() == 1);

  ssize_t retval;
  mode_t mode = S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH;
  int fd      = open(file.c_str(), O_WRONLY | O_CREAT | O_TRUNC, mode);
  mapping mm  = mappings.back();
  mappings.pop_back();

  size_t total = mm.len;
  char* ptr    = (char*)mm.ptr;
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

uint64_t FileGraph::getEdgeIdx(GraphNode src, GraphNode dst) {
  // loop through all neighbors of src, looking for a match with dst
  if (graphVersion == 1) {
    for (auto ii = (uint32_t*)raw_neighbor_begin(src),
              ei = (uint32_t*)raw_neighbor_end(src);
         ii != ei; ++ii) {
      if (convert_le32toh(*ii) == dst)
        return std::distance((uint32_t*)outs, ii);
    }

    return ~static_cast<uint64_t>(0);
  } else if (graphVersion == 2) {
    for (auto ii = (uint64_t*)raw_neighbor_begin(src),
              ei = (uint64_t*)raw_neighbor_end(src);
         ii != ei; ++ii) {
      if (convert_le64toh(*ii) == dst)
        return std::distance((uint64_t*)outs, ii);
    }

    return ~static_cast<uint64_t>(0);
  } else {
    GALOIS_DIE("unknown file version: ", graphVersion);
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

  // different graph version have different edge sizes
  if (graphVersion == 1) {
    edgeSize = sizeof(uint32_t);
  } else if (graphVersion == 2) {
    edgeSize = sizeof(uint64_t);
  } else {
    GALOIS_DIE("unknown file version at pageInByNode", graphVersion);
  }

  // determine which nodes this id is responsible for paging in
  auto r = divideByNode(sizeof(uint64_t), sizeofEdgeData + edgeSize, id, total)
               .first;

  // get begin edge and end edge locations
  // add node offset because edge_begin assumes a global id while divideByNode
  // returns LOCAL ids (same below with edge_end)
  size_t ebegin = *edge_begin(*r.first + nodeOffset);
  size_t eend   = ebegin;

  if (r.first != r.second)
    eend = *edge_end(*r.second - 1 + nodeOffset);

  // page in the outIdx array
  pageInReadOnly(outIdx + *r.first,
                 std::distance(r.first, r.second) * sizeof(*outIdx),
                 runtime::pagePoolSize());

  // page in outs array
  if (graphVersion == 1) {
    pageInReadOnly((uint32_t*)outs + ebegin, (eend - ebegin) * sizeof(uint32_t),
                   runtime::pagePoolSize());
  } else {
    pageInReadOnly((uint64_t*)outs + ebegin, (eend - ebegin) * sizeof(uint64_t),
                   runtime::pagePoolSize());
  }

  // page in edge data
  pageInReadOnly(edgeData + ebegin * sizeofEdgeData,
                 (eend - ebegin) * sizeofEdgeData, runtime::pagePoolSize());
}

void* FileGraph::raw_neighbor_begin(GraphNode N) {
  if (graphVersion == 1) {
    return &(((uint32_t*)outs)[*edge_begin(N)]);
  } else if (graphVersion == 2) {
    return &(((uint64_t*)outs)[*edge_begin(N)]);
  } else {
    GALOIS_DIE("unknown file version: ", graphVersion);
  }

  return nullptr;
}

void* FileGraph::raw_neighbor_end(GraphNode N) {
  if (graphVersion == 1) {
    return &(((uint32_t*)outs)[*edge_end(N)]);
  } else if (graphVersion == 2) {
    return &(((uint64_t*)outs)[*edge_end(N)]);
  } else {
    GALOIS_DIE("unknown file version: ", graphVersion);
  }

  return nullptr;
}

FileGraph::edge_iterator FileGraph::edge_begin(GraphNode N) {
  size_t idx = 0;
  if (N > nodeOffset) {
    numBytesReadIndex += 8;
    idx = std::min(convert_le64toh(outIdx[N - 1 - nodeOffset]),
                   static_cast<uint64_t>(edgeOffset + numEdges)) -
          edgeOffset;
  } else if (N != nodeOffset) {
    printf("WARNING: reading node out of bounds for this file graph\n");
    // TODO die here?
  }
  return edge_iterator(idx);
}

FileGraph::edge_iterator FileGraph::edge_end(GraphNode N) {
  size_t idx = 0;
  if (N >= nodeOffset) {
    numBytesReadIndex += 8;
    idx = std::min(convert_le64toh(outIdx[N - nodeOffset]),
                   static_cast<uint64_t>(edgeOffset + numEdges)) -
          edgeOffset;
  } else {
    printf("WARNING: reading node out of bounds for this file graph\n");
    // TODO die here?
  }
  return edge_iterator(idx);
}

FileGraph::GraphNode FileGraph::getEdgeDst(edge_iterator it) {
  if (graphVersion == 1) {
    numBytesReadEdgeDst += 4;
    // can safely return 32 bit as 64 bit
    return convert_le32toh(((uint32_t*)outs)[*it]);
  } else if (graphVersion == 2) {
    numBytesReadEdgeDst += 8;
    return convert_le64toh(((uint64_t*)outs)[*it]);
  } else {
    GALOIS_DIE("unknown file version: ", graphVersion);
  }

  return -1;
}

FileGraph::node_id_iterator FileGraph::node_id_begin() const {
  return boost::make_transform_iterator(&((uint32_t*)outs)[0], Convert32());
}

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
  return getEdgeIdx(N1, N2) != ~static_cast<uint64_t>(0);
}

FileGraph::iterator FileGraph::begin() const { return iterator(nodeOffset); }

FileGraph::iterator FileGraph::end() const {
  return iterator(nodeOffset + numNodes);
}

void FileGraph::initNodeDegrees() {
  if (!this->node_degrees.size()) {
    // allocate memory
    this->node_degrees.create(this->numNodes);
    // loop over all nodes, calculate degrees
    galois::do_all(
        galois::iterate((uint64_t)0, this->numNodes),
        [&](unsigned long n) {
          // calculate and save degrees
          if (n != 0) {
            this->node_degrees.set(n, this->outIdx[n] - this->outIdx[n - 1]);
          } else {
            this->node_degrees.set(n, this->outIdx[0]);
          }
        },
        galois::loopname("FileGraphInitNodeDegrees"), galois::no_stats());
  }
}

uint64_t FileGraph::getDegree(uint32_t node_id) const {
  // node_degrees array should be initialized
  assert(this->node_degrees.size());
  return this->node_degrees[node_id];
}

void FileGraphWriter::phase1() {
  graphVersion = numNodes <= std::numeric_limits<uint32_t>::max() ? 1 : 2;

  size_t bytes    = galois::graphs::rawBlockSize(numNodes, numEdges, sizeofEdge,
                                                 graphVersion);
  char* mmap_base = reinterpret_cast<char*>(mmap(
      nullptr, bytes, PROT_READ | PROT_WRITE, _MAP_ANON | MAP_PRIVATE, -1, 0));
  if (mmap_base == MAP_FAILED)
    GALOIS_SYS_DIE("failed allocating graph to write");

  mappings.push_back({mmap_base, bytes});

  uint64_t* fptr = reinterpret_cast<uint64_t*>(mmap_base);
  // set header info
  *fptr++    = convert_htole64(graphVersion);
  *fptr++    = convert_htole64(sizeofEdge);
  *fptr++    = convert_htole64(numNodes);
  *fptr++    = convert_htole64(numEdges);
  nodeOffset = 0;
  edgeOffset = 0;
  outIdx     = fptr;

  // move over to outgoing edge data and save it
  fptr += numNodes;
  outs = reinterpret_cast<void*>(fptr);

  // skip memory differently depending on file version
  edgeData = graphVersion == 1
                 ? reinterpret_cast<char*>(reinterpret_cast<uint32_t*>(fptr) +
                                           numEdges + numEdges % 2) // padding
                 : reinterpret_cast<char*>(
                       /*reinterpret_cast<uint64_t*>*/ (fptr) + numEdges);
}

void FileGraphWriter::phase2() {
  if (numNodes == 0)
    return;

  // Turn counts into partial sums
  uint64_t* prev = outIdx;
  for (uint64_t *ii = outIdx + 1, *ei = outIdx + numNodes; ii != ei;
       ++ii, ++prev) {
    *ii += *prev;
  }
  assert(outIdx[numNodes - 1] == numEdges);

  starts = std::make_unique<uint64_t[]>(numNodes);
}

} // namespace graphs
} // namespace galois
