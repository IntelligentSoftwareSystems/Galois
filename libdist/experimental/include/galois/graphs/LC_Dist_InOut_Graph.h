/**
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of XYZ License (a
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

#ifndef GALOIS_GRAPH_LC_DIST_INOUT_H
#define GALOIS_GRAPH_LC_DIST_INOUT_H

#include <vector>
#include <iostream>

#include "galois/runtime/Context.h"

namespace galois {
namespace graphs {

template <typename NodeTy, typename EdgeTy>
class LC_Dist_InOut {

  struct EdgeImplTy;

  struct NodeImplTy : public runtime::Lockable {
    NodeImplTy(EdgeImplTy* start, unsigned len, EdgeImplTy* In_start,
               unsigned len_inEdges)
        : b(start), e(start), len(len), b_inEdges(In_start),
          e_inEdges(In_start), len_inEdges(len_inEdges), remote(false) {}
    ~NodeImplTy() {
      if (remote) {
        delete[] b;
        delete[] b_inEdges;
      }
      b = e     = nullptr;
      b_inEdges = e_inEdges = nullptr;
    }

    // Serialization support
    typedef int tt_has_serialize;
    NodeImplTy(runtime::DeSerializeBuffer& buf) : remote(true) {
      ptrdiff_t diff;
      ptrdiff_t diff_inEdges;
      gDeserialize(buf, data, len, diff, len_inEdges, diff_inEdges);
      b = e     = len ? (new EdgeImplTy[len]) : nullptr;
      b_inEdges = e_inEdges =
          len_inEdges ? (new EdgeImplTy[len_inEdges]) : nullptr;
      while (diff--) {
        runtime::gptr<NodeImplTy> ptr;
        EdgeTy tmp;
        gDeserialize(buf, ptr, tmp);
        append(ptr, tmp);
      }

      while (diff_inEdges--) {
        runtime::gptr<NodeImplTy> ptr;
        EdgeTy tmp;
        gDeserialize(buf, ptr, tmp);
        append_inEdges(ptr, tmp);
      }
    }

    void serialize(runtime::SerializeBuffer& buf) const {
      EdgeImplTy* begin = b;
      EdgeImplTy* end   = e;

      EdgeImplTy* begin_In = b_inEdges;
      EdgeImplTy* end_In   = e_inEdges;

      ptrdiff_t diff         = end - begin;
      ptrdiff_t diff_inEdges = end_In - begin_In;
      gSerialize(buf, data, len, diff, len_inEdges, diff_inEdges);
      while (begin != end) {
        gSerialize(buf, begin->dst, begin->data);
        ++begin;
      }
      while (begin_In != end_In) {
        gSerialize(buf, begin_In->dst, begin_In->data);
        ++begin_In;
      }
    }
    void deserialize(runtime::DeSerializeBuffer& buf) {
      assert(!remote);
      ptrdiff_t diff;
      unsigned _len;
      ptrdiff_t diff_inEdges;
      unsigned _len_inEdges;

      EdgeImplTy* begin    = b;
      EdgeImplTy* begin_In = b_inEdges;
      gDeserialize(buf, data, _len, diff, _len_inEdges, diff_inEdges);
      assert(_len == len);
      assert(diff >= e - b);
      // inEdges
      assert(_len_inEdges == len_inEdges);
      assert(diff_inEdges >= e_inEdges - b_inEdges);
      while (diff--) {
        gDeserialize(buf, begin->dst, begin->data);
        ++begin;
      }
      e = begin;
      while (diff_inEdges--) {
        gDeserialize(buf, begin_In->dst, begin_In->data);
        ++begin_In;
      }
      e_inEdges = begin_In;
    }

    NodeTy data;
    EdgeImplTy* b;
    EdgeImplTy* e;
    unsigned len;
    // InEdge list for each node
    EdgeImplTy* b_inEdges;
    EdgeImplTy* e_inEdges;
    unsigned len_inEdges;
    //
    bool remote;

    EdgeImplTy* append(runtime::gptr<NodeImplTy> dst, const EdgeTy& data) {
      assert(e - b < len);
      e->dst  = dst;
      e->data = data;
      return e++;
    }

    EdgeImplTy* append(runtime::gptr<NodeImplTy> dst) {
      assert(e - b < len);
      e->dst = dst;
      return e++;
    }

    // Appending In_Edges
    EdgeImplTy* append_inEdges(runtime::gptr<NodeImplTy> dst,
                               const EdgeTy& data) {
      assert(e_inEdges - b_inEdges < len_inEdges);
      e_inEdges->dst  = dst;
      e_inEdges->data = data;
      return e_inEdges++;
    }

    EdgeImplTy* append_inEdges(runtime::gptr<NodeImplTy> dst) {
      assert(e_inEdges - b_inEdges < len_inEdges);
      e_inEdges->dst = dst;
      return e_inEdges++;
    }

    EdgeImplTy* begin() { return b; }
    EdgeImplTy* end() { return e; }

    // for In_edge list for each node
    EdgeImplTy* begin_inEdges() { return b_inEdges; }
    EdgeImplTy* end_inEdges() { return e_inEdges; }
  };

  struct EdgeImplTy {
    runtime::gptr<NodeImplTy> dst;
    EdgeTy data;
  };

public:
  typedef runtime::gptr<NodeImplTy> GraphNode;
  // G
  template <bool _has_id>
  struct with_id {
    typedef LC_Dist_InOut type;
  };

  template <typename _node_data>
  struct with_node_data {
    typedef LC_Dist_InOut<_node_data, EdgeTy> type;
  };

  template <typename _edge_data>
  struct with_edge_data {
    typedef LC_Dist_InOut<NodeTy, _edge_data> type;
  };

private:
  typedef std::vector<NodeImplTy> NodeData;
  typedef std::vector<EdgeImplTy> EdgeData;
  NodeData Nodes;
  EdgeData Edges;
  EdgeData In_Edges;

  std::vector<std::pair<GraphNode, std::atomic<int>>> Starts;
  std::vector<unsigned> Num;
  std::vector<unsigned> PrefixNum;

  runtime::PerHost<LC_Dist_InOut> self;

  friend class runtime::PerHost<LC_Dist_InOut>;

  LC_Dist_InOut(runtime::PerHost<LC_Dist_InOut> _self,
                std::vector<unsigned>& edges, std::vector<unsigned>& In_edges)
      : Starts(runtime::NetworkInterface::Num),
        Num(runtime::NetworkInterface::Num),
        PrefixNum(runtime::NetworkInterface::Num), self(_self) {
    // std::cerr << runtime::NetworkInterface::ID << " Construct\n";
    runtime::trace("LC_Dist_InOut with % nodes total\n", edges.size());
    // Fill up metadata vectors
    for (unsigned h = 0; h < runtime::NetworkInterface::Num; ++h) {
      auto p = block_range(edges.begin(), edges.end(), h,
                           runtime::NetworkInterface::Num);
      Num[h] = std::distance(p.first, p.second);
    }
    std::partial_sum(Num.begin(), Num.end(), PrefixNum.begin());

    // std::copy(Num.begin(), Num.end(),
    // std::ostream_iterator<unsigned>(std::cout, ",")); std::cout << "\n";
    // std::copy(PrefixNum.begin(), PrefixNum.end(),
    // std::ostream_iterator<unsigned>(std::cout, ",")); std::cout << "\n";

    // Block nodes
    auto p =
        block_range(edges.begin(), edges.end(), runtime::NetworkInterface::ID,
                    runtime::NetworkInterface::Num);
    // Block in_edge vector
    auto p_inEdges = block_range(In_edges.begin(), In_edges.end(),
                                 runtime::NetworkInterface::ID,
                                 runtime::NetworkInterface::Num);

    // NOTE: range of p and p_inEdges will be same since the size(edges) ==
    // size(In_edges) Allocate Edges
    unsigned sum = std::accumulate(
        p.first, p.second, 0); // Total number of egdes on this host (me)
    Edges.resize(sum);
    EdgeImplTy* cur = &Edges[0];

    // Allocate In_Edges
    unsigned sum_inEdges =
        std::accumulate(p_inEdges.first, p_inEdges.second, 0);
    In_Edges.resize(sum_inEdges);
    EdgeImplTy* cur_In = &In_Edges[0]; // cur edgelist for in_edges

    // allocate Nodes
    Nodes.reserve(std::distance(p.first, p.second));
    for (auto ii = p.first, ii_in = p_inEdges.first; ii != p.second;
         ++ii, ++ii_in) {
      Nodes.emplace_back(cur, *ii, cur_In, *ii_in);
      cur += *ii;
      cur_In += *ii_in;
    }
    Starts[runtime::NetworkInterface::ID].first  = GraphNode(&Nodes[0]);
    Starts[runtime::NetworkInterface::ID].second = 2;
  }

  ~LC_Dist_InOut() {
    auto ii =
        iterator(this, runtime::NetworkInterface::ID == 0
                           ? 0
                           : PrefixNum[runtime::NetworkInterface::ID - 1]);
    auto ee = iterator(this, PrefixNum[runtime::NetworkInterface::ID]);
    for (; ii != ee; ++ii)
      acquireNode(*ii, MethodFlag::ALL);
  }

  static void getStart(runtime::PerHost<LC_Dist_InOut> graph, uint32_t whom) {
    // std::cerr << runtime::NetworkInterface::ID << " getStart " << whom <<
    // "\n";
    runtime::getSystemNetworkInterface().sendAlt(
        whom, putStart, graph, runtime::NetworkInterface::ID,
        graph->Starts[runtime::NetworkInterface::ID].first);
  }

  static void putStart(runtime::PerHost<LC_Dist_InOut> graph, uint32_t whom,
                       GraphNode start) {
    std::cerr << runtime::NetworkInterface::ID << " putStart " << whom << "\n";
    graph->Starts[whom].first  = start;
    graph->Starts[whom].second = 2;
  }

  // blocking
  void fillStarts(uint32_t host) {
    if (Starts[host].second != 2) {
      int ex    = 0;
      bool swap = Starts[host].second.compare_exchange_strong(ex, 1);
      if (swap)
        runtime::getSystemNetworkInterface().sendAlt(
            host, getStart, self, runtime::NetworkInterface::ID);
      while (Starts[host].second != 2) {
        runtime::doNetworkWork();
      }
    }
    assert(Starts[host].first);
  }

  GraphNode generateNodePtr(unsigned x) {
    auto ii = std::upper_bound(PrefixNum.begin(), PrefixNum.end(), x);
    assert(ii != PrefixNum.end());
    unsigned host   = std::distance(PrefixNum.begin(), ii);
    unsigned offset = x - (host == 0 ? 0 : *(ii - 1));
    fillStarts(host);
    return Starts[host].first + offset;
  }

  void acquireNode(runtime::gptr<NodeImplTy> node, galois::MethodFlag mflag) {
    acquire(node, mflag);
  }

public:
  typedef EdgeImplTy edge_data_type;
  typedef NodeImplTy node_data_type;
  typedef typename EdgeData::reference edge_data_reference;
  typedef typename NodeData::reference node_data_reference;

  // This is technically a const (non-mutable) iterator
  class iterator
      : public std::iterator<std::random_access_iterator_tag, GraphNode,
                             ptrdiff_t, GraphNode, GraphNode> {
    LC_Dist_InOut* g;
    unsigned x;

    friend class LC_Dist_InOut;
    iterator(LC_Dist_InOut* _g, unsigned _x) : g(_g), x(_x) {}

  public:
    iterator() : g(nullptr), x(0) {}
    iterator(const iterator&) = default;
    bool operator==(const iterator& rhs) const {
      return g == rhs.g & x == rhs.x;
    }
    bool operator!=(const iterator& rhs) const { return !(*this == rhs); }
    bool operator<(const iterator& rhs) const { return x < rhs.x; }
    bool operator>(const iterator& rhs) const { return x > rhs.x; }
    bool operator<=(const iterator& rhs) const { return x <= rhs.x; }
    bool operator>=(const iterator& rhs) const { return x >= rhs.x; }

    GraphNode operator*() { return g->generateNodePtr(x); }
    GraphNode operator->() { return g->generateNodePtr(x); }
    GraphNode operator[](int n) { return g->generateNodePtr(x + n); }

    iterator& operator++() {
      ++x;
      return *this;
    }
    iterator& operator--() {
      --x;
      return *this;
    }
    iterator operator++(int) {
      auto old = *this;
      ++x;
      return old;
    }
    iterator operator--(int) {
      auto old = *this;
      --x;
      return old;
    }
    iterator operator+(int n) {
      auto ret = *this;
      ret.x += n;
      return ret;
    }
    iterator operator-(int n) {
      auto ret = *this;
      ret.x -= n;
      return ret;
    }
    iterator& operator+=(int n) {
      x += n;
      return *this;
    }
    iterator& operator-=(int n) {
      x -= n;
      return *this;
    }
    ptrdiff_t operator-(const iterator& rhs) { return x - rhs.x; }
    // Trivially copyable
    typedef int tt_is_copyable;
  };

  typedef iterator local_iterator;
  typedef EdgeImplTy* edge_iterator;

  //! Creation and destruction
  typedef runtime::PerHost<LC_Dist_InOut> pointer;
  static pointer allocate(std::vector<unsigned>& edges,
                          std::vector<unsigned>& In_edges) {
    return pointer::allocate(edges, In_edges);
  }
  static void deallocate(pointer ptr) { pointer::deallocate(ptr); }

  //! Mutation

  template <typename... Args>
  edge_iterator addEdge(GraphNode src, GraphNode dst, const EdgeTy& data,
                        MethodFlag mflag = MethodFlag::ALL) {
    acquire(src, mflag);
    src->append(dst, data);
    // return src->append(dst, data);
  }

  edge_iterator addEdge(GraphNode src, GraphNode dst,
                        MethodFlag mflag = MethodFlag::ALL) {
    acquire(src, mflag);
    src->append(dst);
    // return src->append(dst);
  }

  // Adding inEdges to the Graph.
  template <typename... Args>
  edge_iterator addInEdge(GraphNode src, GraphNode dst, const EdgeTy& data,
                          MethodFlag mflag = MethodFlag::ALL) {
    acquire(src, mflag);
    return src->append_inEdges(dst, data);
  }

  edge_iterator addInEdge(GraphNode src, GraphNode dst,
                          MethodFlag mflag = MethodFlag::ALL) {
    acquire(src, mflag);
    return src->append_inEdges(dst);
  }
  //! Access

  NodeTy& at(GraphNode N, MethodFlag mflag = MethodFlag::ALL) {
    acquire(N, mflag);
    return N->data;
  }

  EdgeTy& at(edge_iterator E, MethodFlag mflag = MethodFlag::ALL) {
    return E->data;
  }

  GraphNode dst(edge_iterator E, MethodFlag mflag = MethodFlag::ALL) {
    return E->dst;
  }

  //! Capacity
  unsigned size() { return *PrefixNum.rbegin(); }

  //! Iterators

  iterator begin() { return iterator(this, 0); }
  iterator end() {
    return iterator(this, PrefixNum[runtime::NetworkInterface::Num - 1]);
  }

  local_iterator local_begin() {
    if (runtime::LL::getTID() == 0)
      return iterator(this, runtime::NetworkInterface::ID == 0
                                ? 0
                                : PrefixNum[runtime::NetworkInterface::ID - 1]);
    else
      return local_end();
  }
  local_iterator local_end() {
    return iterator(this, PrefixNum[runtime::NetworkInterface::ID]);
  }

  edge_iterator edge_begin(GraphNode N, MethodFlag mflag = MethodFlag::ALL) {
    acquire(N, mflag);
    if (mflag != MethodFlag::SRC_ONLY && mflag != MethodFlag::NONE)

      /** prefetch **/
      for (auto ii = N->begin(), ee = N->end(); ii != ee; ++ii) {
        prefetch(ii->dst);
      }
    for (edge_iterator ii = N->begin(), ee = N->end(); ii != ee; ++ii) {
      acquireNode(ii->dst, mflag);
    }
    return N->begin();
  }

  edge_iterator edge_end(GraphNode N, MethodFlag mflag = MethodFlag::ALL) {
    acquireNode(N, mflag);
    return N->end();
  }

  /**
   * In Edge iterators
   *
   */

  edge_iterator in_edge_begin(GraphNode N, MethodFlag mflag = MethodFlag::ALL) {
    acquire(N, mflag);
    if (mflag != MethodFlag::SRC_ONLY && mflag != MethodFlag::NONE)

      /** prefetch **/
      for (auto ii = N->begin_inEdges(), ee = N->end_inEdges(); ii != ee;
           ++ii) {
        prefetch(ii->dst);
      }
    for (edge_iterator ii = N->begin_inEdges(), ee = N->end_inEdges(); ii != ee;
         ++ii) {
      acquireNode(ii->dst, mflag);
    }

    return N->begin_inEdges();
  }

  edge_iterator in_edge_end(GraphNode N, MethodFlag mflag = MethodFlag::ALL) {
    acquireNode(N, mflag);
    return N->end_inEdges();
  }

  template <typename Compare>
  void sort_edges(GraphNode N, Compare comp,
                  MethodFlag mflag = MethodFlag::ALL) {
    std::sort(edge_begin(N, mflag), edge_end(N, mflag),
              [&comp](const EdgeImplTy& e1, const EdgeImplTy& e2) {
                return comp(e1.dst, e1.data, e2.dst, e2.data);
              });
  }

  /*** Additional functions ***/
  void prefetch_all() {
    for (auto ii = this->begin(); ii != this->end(); ++ii) {
      prefetch(ii);
    }
  }
};

} // namespace graphs
} // namespace galois

#endif
