/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
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

#include "galois/graphs/OCGraph.h"
#include "galois/runtime/Mem.h"
#include "galois/gIO.h"

#include <cassert>

#include <fcntl.h>
#ifdef __linux__
#include <linux/mman.h>
#endif
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>

using namespace galois::graphs;

// File format V1:
// version (1) {uint64_t LE}
// EdgeType size {uint64_t LE}
// numNodes {uint64_t LE}
// numEdges {uint64_t LE}
// outindexs[numNodes] {uint64_t LE} (outindex[nodeid] is index of first edge
// for nodeid + 1 (end interator.  node 0 has an implicit start iterator of 0.
// outedges[numEdges] {uint32_t LE}
// potential padding (32bit max) to Re-Align to 64bits
// EdgeType[numEdges] {EdgeType size}

#ifdef HAVE_MMAP64
template <typename... Args>
void* mmap_big(Args... args) {
  return mmap64(std::forward<Args>(args)...);
}
#else
template <typename... Args>
void* mmap_big(Args... args) {
  return mmap(std::forward<Args>(args)...);
}
#endif

OCFileGraph::~OCFileGraph() {
  if (masterMapping)
    munmap(masterMapping, masterLength);
  if (masterFD != -1)
    close(masterFD);
}

void OCFileGraph::Block::unload() {
  if (!m_mapping)
    return;

  if (munmap(m_mapping, m_length) != 0) {
    GALOIS_SYS_DIE("failed unallocating");
  }
  m_mapping = 0;
}

void OCFileGraph::Block::load(int fd, offset_t offset, size_t begin, size_t len,
                              size_t sizeof_data) {
  assert(m_mapping == 0);

  offset_t start = offset + begin * sizeof_data;
  offset_t aligned =
      start & ~static_cast<offset_t>(galois::runtime::pagePoolSize() - 1);

  int _MAP_BASE = MAP_PRIVATE;
#ifdef MAP_POPULATE
  _MAP_BASE |= MAP_POPULATE;
#endif
  m_length =
      len * sizeof_data +
      galois::runtime::pagePoolSize(); // account for round off due to alignment
  m_mapping = mmap_big(nullptr, m_length, PROT_READ, _MAP_BASE, fd, aligned);
  if (m_mapping == MAP_FAILED) {
    GALOIS_SYS_DIE("failed allocating ", fd);
  }

  m_data = reinterpret_cast<char*>(m_mapping);
  assert(aligned <= start);
  assert(start - aligned <=
         static_cast<offset_t>(galois::runtime::pagePoolSize()));
  m_data += start - aligned;
  m_begin       = begin;
  m_sizeof_data = sizeof_data;
}

void OCFileGraph::load(segment_type& s, edge_iterator begin, edge_iterator end,
                       size_t sizeof_data) {
  size_t bb  = *begin;
  size_t len = *end - *begin;

  offset_t outs = (4 + numNodes) * sizeof(uint64_t);
  offset_t data = outs + (numEdges + (numEdges & 1)) * sizeof(uint32_t);

  s.outs.load(masterFD, outs, bb, len, sizeof(uint32_t));
  if (sizeof_data)
    s.edgeData.load(masterFD, data, bb, len, sizeof_data);

  s.loaded = true;
}

static void readHeader(int fd, uint64_t& numNodes, uint64_t& numEdges) {
  void* m = mmap(0, 4 * sizeof(uint64_t), PROT_READ, MAP_PRIVATE, fd, 0);
  if (m == MAP_FAILED) {
    GALOIS_SYS_DIE("failed reading ", fd);
  }

  uint64_t* ptr = reinterpret_cast<uint64_t*>(m);
  assert(ptr[0] == 1);
  numNodes = ptr[2];
  numEdges = ptr[3];

  if (munmap(m, 4 * sizeof(uint64_t))) {
    GALOIS_SYS_DIE("failed reading ", fd);
  }
}

void OCFileGraph::fromFile(const std::string& filename) {
  masterFD = open(filename.c_str(), O_RDONLY);
  if (masterFD == -1) {
    GALOIS_SYS_DIE("failed opening ", filename);
  }

  readHeader(masterFD, numNodes, numEdges);
  masterLength  = 4 * sizeof(uint64_t) + numNodes * sizeof(uint64_t);
  int _MAP_BASE = MAP_PRIVATE;
#ifdef MAP_POPULATE
  _MAP_BASE |= MAP_POPULATE;
#endif
  masterMapping = mmap(0, masterLength, PROT_READ, _MAP_BASE, masterFD, 0);
  if (masterMapping == MAP_FAILED) {
    GALOIS_SYS_DIE("failed reading ", filename);
  }

  outIdx = reinterpret_cast<uint64_t*>(masterMapping);
  outIdx += 4;
}
