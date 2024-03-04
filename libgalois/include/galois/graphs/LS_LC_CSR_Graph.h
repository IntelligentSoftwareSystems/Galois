/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2024, The University of Texas at Austin. All rights reserved.
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

#ifndef GALOIS_GRAPHS_LC_CSR_GRAPH_H
#define GALOIS_GRAPHS_LC_CSR_GRAPH_H

#include <unordered_set>
#include <iterator>
#include <cstddef>

#include <boost/range/iterator_range_core.hpp>
#include <boost/range/counting_range.hpp>
#include <boost/iterator/iterator_facade.hpp>

#include "galois/config.h"
#include "galois/LargeVector.h"

namespace galois::graphs {

/**
 * Local computation graph.
 */
template <bool concurrent = true>
class LS_LC_CSR_Graph : private boost::noncopyable {
public:
  using VertexTopologyID = uint64_t;
  using VertexRange =
      boost::iterator_range<boost::counting_iterator<VertexTopologyID>>;

  struct EdgeHandle {
  private:
    uint8_t buffer : 1;
    uint64_t index : 48;

    EdgeHandle(uint8_t buffer, uint64_t index) : buffer(buffer), index(index) {}

    EdgeHandle(uint64_t const& v) : buffer(v >> 63), index(v) {}

    friend class LS_LC_CSR_Graph;

  public:
    EdgeHandle(EdgeHandle const&) = default;
    EdgeHandle(EdgeHandle&&)      = default;

  } __attribute__((packed));

private:
  using SpinLock = galois::substrate::PaddedLock<concurrent>;

  // forward-declarations
  struct VertexMetadata;
  struct EdgeMetadata;

  class EdgeIterator;
  using EdgeRange = boost::iterator_range<EdgeIterator>;

  std::vector<VertexMetadata> m_vertices;

  LargeVector<EdgeMetadata> m_edges[2];

  // To avoid deadlock between updates and compaction, at least one vertex lock
  // must be held to acquire m_edges_lock.
  SpinLock m_edges_lock;

  // returns a reference to the metadata for the pointed-to edge
  inline EdgeMetadata& getEdgeMetadata(EdgeHandle const& handle) {
    return getEdgeMetadata(handle.buffer, handle.index);
  }

  inline EdgeMetadata& getEdgeMetadata(uint8_t buffer, uint64_t index) const {
    return m_edges[buffer][index];
  }

public:
  LS_LC_CSR_Graph(uint64_t num_vertices)
      : m_vertices(num_vertices, VertexMetadata()) {}

  inline uint64_t size() const noexcept { return m_vertices.size(); }

  inline VertexTopologyID begin() const noexcept {
    return static_cast<VertexTopologyID>(0);
  }

  inline VertexTopologyID end() const noexcept {
    return static_cast<VertexTopologyID>(m_vertices.size());
  }

  VertexRange vertices() { return VertexRange(begin(), end()); }

  EdgeRange edges(VertexTopologyID node) {
    auto& vertex_meta = m_vertices[node];
    auto const* ii    = &getEdgeMetadata(vertex_meta.buffer, vertex_meta.begin);
    auto const* ee    = &getEdgeMetadata(vertex_meta.buffer, vertex_meta.end);

    return EdgeRange(EdgeIterator(ii, ee), EdgeIterator(ee, ee));
  }

  int addEdgesTopologyOnly(VertexTopologyID src,
                           const std::vector<VertexTopologyID> dsts) {
    auto& vertex_meta = m_vertices[src];

    // Copies the edge list to the end of m_edges[1], prepending
    // the new edges.

    vertex_meta.lock();
    {
      uint64_t const new_degree = vertex_meta.degree + dsts.size();
      uint64_t new_begin;
      m_edges_lock.lock();
      {
        new_begin = m_edges[1].size();
        m_edges[1].resize(new_begin + new_degree);
      }
      m_edges_lock.unlock();
      uint64_t const new_end = new_begin + new_degree;

      // insert new edges
      std::transform(dsts.begin(), dsts.end(), &getEdgeMetadata(1, new_begin),
                     [](VertexTopologyID dst) {
                       return EdgeMetadata{.flags = 0, .dst = dst};
                     });

      // copy old, non-tombstoned edges
      std::copy_if(&getEdgeMetadata(vertex_meta.buffer, vertex_meta.begin),
                   &getEdgeMetadata(vertex_meta.buffer, vertex_meta.end),
                   &getEdgeMetadata(1, new_begin + dsts.size()),
                   [](EdgeMetadata& edge) { return !edge.is_tomb(); });

      // update vertex metadata
      vertex_meta.buffer = 1;
      vertex_meta.begin  = new_begin;
      vertex_meta.end    = new_end;
      vertex_meta.degree += dsts.size();
    }
    vertex_meta.unlock();

    return 0;
  }

  int deleteEdges(VertexTopologyID src,
                  const std::vector<VertexTopologyID>& edges) {
    std::unordered_set<VertexTopologyID> edges_set(edges.begin(), edges.end());

    auto& vertex_meta = m_vertices[src];
    vertex_meta.lock();
    {
      for (auto i = vertex_meta.begin; i < vertex_meta.end; ++i) {
        EdgeMetadata& edge_meta =
            getEdgeMetadata(EdgeHandle(vertex_meta.buffer, i));
        if (!edge_meta.is_tomb() &&
            edges_set.find(edge_meta.dst) != edges_set.end()) {
          edge_meta.tomb();
          --vertex_meta.degree;
          // remove tombstoned edges from the start of the edge list
          if (i == vertex_meta.begin)
            ++vertex_meta.begin;
        }
      }

      // remove tombstoned edges from the end of the edge list
      for (auto i = vertex_meta.end; i > vertex_meta.begin; --i) {
        if (getEdgeMetadata(EdgeHandle(vertex_meta.buffer, i - 1)).is_tomb()) {
          --vertex_meta.end;
          --vertex_meta.degree;
        } else {
          break;
        }
      }
    }
    vertex_meta.unlock();

    return 0;
  }

  VertexTopologyID getEdgeDst(EdgeHandle edge) {
    return getEdgeMetadata(edge).dst;
  }

  // Performs the compaction algorithm by copying any vertices left in buffer 0
  // to buffer 1, then swapping the buffers.
  //
  // Should not be called from within a Galois parallel kernel.
  void compact() {
    using std::swap;

    // move from buffer 0 to buffer 1
    galois::do_all(galois::iterate(vertices().begin(), vertices().end()),
                   [&](VertexTopologyID vertex_id) {
                     VertexMetadata& vertex_meta = m_vertices[vertex_id];
                     vertex_meta.lock();

                     if (vertex_meta.buffer == 0) {
                       uint64_t new_begin;
                       m_edges_lock.lock();
                       {
                         new_begin = m_edges[1].size();
                         m_edges[1].resize(new_begin + vertex_meta.degree);
                       }
                       m_edges_lock.unlock();

                       uint64_t new_end = new_begin + vertex_meta.degree;

                       std::copy_if(&getEdgeMetadata(0, vertex_meta.begin),
                                    &getEdgeMetadata(0, vertex_meta.end),
                                    &getEdgeMetadata(1, new_begin),
                                    [](EdgeMetadata& edge_meta) {
                                      return !edge_meta.is_tomb();
                                    });

                       vertex_meta.begin = new_begin;
                       vertex_meta.end   = new_end;
                     }

                     // we are about to swap the buffers, so all vertices will
                     // be in buffer 0
                     vertex_meta.buffer = 0;

                     // don't release the vertex lock until after the edge
                     // arrays are swapped
                   });

    // At this point, there are no more live edges in buffer 0.
    // We also hold the lock for all vertices, so nobody else can hold
    // m_edges_lock.
    m_edges_lock.lock();
    {
      m_edges[0].resize(0);
      swap(m_edges[0], m_edges[1]);
    }
    m_edges_lock.unlock();

    galois::do_all(
        galois::iterate(vertices().begin(), vertices().end()),
        [&](VertexTopologyID vertex_id) { m_vertices[vertex_id].unlock(); });
  }

private:
  struct VertexMetadata : public SpinLock {
    uint8_t buffer : 1;
    uint64_t begin : 48; // inclusive
    uint64_t end : 48;   // exclusive
    uint64_t degree;

    VertexMetadata() : buffer(0), begin(0), end(0), degree(0) {}

    VertexMetadata(VertexMetadata const& other)
        : buffer(other.buffer), begin(other.begin), end(other.end),
          degree(other.degree) {}

    VertexMetadata(VertexMetadata&& other)
        : buffer(std::move(other.buffer)), begin(std::move(other.begin)),
          end(std::move(other.end)), degree(std::move(other.degree)) {}
  };

  struct EdgeMetadata {
    enum Flags : uint16_t { TOMB = 0x1 };

    uint16_t flags : 16;
    VertexTopologyID dst : 48;

    bool is_tomb() const noexcept { return (flags & TOMB) > 0; }
    void tomb() { flags |= TOMB; }
  } __attribute__((packed));

  static_assert(sizeof(EdgeMetadata) <= sizeof(uint64_t));

  class EdgeIterator
      : public boost::iterator_facade<EdgeIterator, EdgeMetadata const,
                                      boost::forward_traversal_tag,
                                      VertexTopologyID> {
  private:
    EdgeMetadata const* m_ptr;
    EdgeMetadata const* const m_end;

    explicit EdgeIterator(EdgeMetadata const* ptr, EdgeMetadata const* end)
        : m_ptr(ptr), m_end(end) {}

    void increment() {
      while (++m_ptr < m_end && m_ptr->is_tomb())
        ;
    };

    // note: equality fails across generations
    bool equal(EdgeIterator const& other) const { return m_ptr == other.m_ptr; }

    VertexTopologyID dereference() const { return m_ptr->dst; }

    friend class LS_LC_CSR_Graph;
    friend class boost::iterator_core_access;
  };
};

}; // namespace galois::graphs

#endif
