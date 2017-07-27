/** Distributed LC InOut Graph -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2012, The University of Texas at Austin. All rights reserved.
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
 * @author Andrew Lenharth <andrewl@lenharth.org>
 * @author Gurbinder Gill <gill@cs.utexas.edu>
 */

#ifndef GALOIS_GRAPH_LC_DIST_INOUT_H
#define GALOIS_GRAPH_LC_DIST_INOUT_H

#include <vector>
#include <iostream>

#include "Galois/Runtime/Context.h"
#include "Galois/Graph/FileGraph.h"
#include "Galois/Runtime/Serialize.h"
#include "Galois/Runtime/PerHostStorage.h"
#include "Galois/Runtime/DistSupport.h"


namespace Galois {
namespace Graph {

template<typename NodeTy>
class LC_Dist_InOut {
  
  struct EdgeImplTy;
  
  struct NodeImplTy :public Runtime::Lockable {
    NodeImplTy(EdgeImplTy* start, unsigned len, EdgeImplTy* In_start, unsigned len_inEdges) :b(start), e(start), len(len), b_inEdges(In_start), e_inEdges(In_start), len_inEdges(len_inEdges), remote(false)
    {}
    ~NodeImplTy() {
      if (remote) {
        delete[] b;
        delete[] b_inEdges;
      }
      b = e = nullptr;
      b_inEdges = e_inEdges = nullptr;
    }
    
    //Serialization support
    typedef int tt_has_serialize;
    NodeImplTy(Runtime::DeSerializeBuffer& buf) :remote(true) {
      ptrdiff_t diff;
      ptrdiff_t diff_inEdges;
      gDeserialize(buf, data, len, diff, len_inEdges, diff_inEdges);
      b = e = len ? (new EdgeImplTy[len]) : nullptr;
      b_inEdges = e_inEdges = len_inEdges ? (new EdgeImplTy[len_inEdges]) : nullptr;
      while (diff--) {
        uint64_t ptr;
        gDeserialize(buf, ptr);
        append(ptr);
      }
      
      while (diff_inEdges--) {
        uint64_t ptr;
        gDeserialize(buf, ptr);
        append_inEdges(ptr);
        
      }
    }
    
    void serialize(Runtime::SerializeBuffer& buf) const {
      EdgeImplTy* begin = b;
      EdgeImplTy* end = e;
      
      EdgeImplTy* begin_In = b_inEdges;
      EdgeImplTy* end_In = e_inEdges;
      
      ptrdiff_t diff = end - begin;
      ptrdiff_t diff_inEdges = end_In - begin_In;
      gSerialize(buf, data, len, diff, len_inEdges, diff_inEdges);
      while (begin != end) {
        gSerialize(buf, begin->dst);
        ++begin;
      }
      while (begin_In != end_In) {
        gSerialize(buf, begin_In->dst);
        ++begin_In;
      }
      
    }
    void deserialize(Runtime::DeSerializeBuffer& buf) {
      assert(!remote);
      ptrdiff_t diff;
      unsigned _len;
      ptrdiff_t diff_inEdges;
      unsigned _len_inEdges;
      
      EdgeImplTy* begin = b;
      EdgeImplTy* begin_In = b_inEdges;
      gDeserialize(buf, data, _len, diff, _len_inEdges, diff_inEdges);
      assert(_len == len);
      assert(diff >= e - b);
      // inEdges
      assert(_len_inEdges == len_inEdges);
      assert(diff_inEdges >= e_inEdges - b_inEdges);
      while (diff--) {
        gDeserialize(buf, begin->dst);
        ++begin;
      }
      e = begin;
      while (diff_inEdges--) {
        gDeserialize(buf, begin_In->dst);
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
    
    EdgeImplTy* append(uint64_t dst) {
      assert(e-b < len);
      e->dst = dst;
      return e++;
    }
    
    // Appending In_Edges
    EdgeImplTy* append_inEdges(uint64_t dst) {
      assert(e_inEdges-b_inEdges < len_inEdges);
      e_inEdges->dst = dst;
      return e_inEdges++;
    }
    
    unsigned get_num_outEdges()
    {
      return len;
    }
    unsigned get_num_inEdges()
    {
      return len_inEdges;
    }
    
    
    EdgeImplTy* begin() { return b; }
    EdgeImplTy* end() { return e; }
    
    // for In_edge list for each node 
    EdgeImplTy* begin_inEdges() { return b_inEdges; }
    EdgeImplTy* end_inEdges() { return e_inEdges; }
    
  };
  
  struct EdgeImplTy {
    uint64_t dst;
  };
  
public:
  
  typedef uint64_t GraphNode;
  //G
  template<bool _has_id>
  struct with_id { typedef LC_Dist_InOut type; };
  
  template<typename _node_data>
  struct with_node_data { typedef LC_Dist_InOut<_node_data> type; };
  
private:

  typedef Runtime::gptr<NodeImplTy> NodePtr;
  
  typedef std::vector<NodeImplTy> NodeData;
  typedef std::vector<EdgeImplTy> EdgeData;
  NodeData Nodes;
  EdgeData Edges;
  EdgeData In_Edges;
  
  std::vector<std::pair<NodePtr, std::atomic<int> > > Starts;
  std::vector<unsigned> Num;
  std::vector<unsigned> PrefixNum;
  
  Runtime::PerHost<LC_Dist_InOut> self;
  
  friend class Runtime::PerHost<LC_Dist_InOut>;
  
  LC_Dist_InOut(Runtime::PerHost<LC_Dist_InOut> _self, std::string inGr, std::string inGrTrans)
    :Starts(Runtime::NetworkInterface::Num), Num(Runtime::NetworkInterface::Num), 
     PrefixNum(Runtime::NetworkInterface::Num), self(_self)
  {
    Galois::Graph::FileGraph fg, fgt;
    fg.fromFile(inGr);
    fgt.fromFile(inGrTrans);
    
    for (unsigned h = 0; h < Runtime::NetworkInterface::Num; ++h) {
      auto p = block_range(fg.begin(), fg.end(), h,
                           Runtime::NetworkInterface::Num);
      Num[h] = std::distance(p.first, p.second);
    }
    std::partial_sum(Num.begin(), Num.end(), PrefixNum.begin());
    
    auto p = block_range(fg.begin(), fg.end(), Runtime::NetworkInterface::ID, Runtime::NetworkInterface::Num);
    auto pt = block_range(fgt.begin(), fgt.end(), Runtime::NetworkInterface::ID, Runtime::NetworkInterface::Num);
    
    //number of nodes on this host
    unsigned num = std::distance(p.first, p.second);
    std::cout << num << " " << std::distance(fg.begin(), fg.end()) << "\n";
    //number of out-edges on this host
    unsigned num_out = std::accumulate(p.first, p.second, 0, [&fg] (unsigned d, const uint64_t& node) { return d + std::distance(fg.edge_begin(node), fg.edge_end(node)); });
    //number of in-edges on this host
    unsigned num_in = std::accumulate(pt.first, pt.second, 0, [&fgt] ( unsigned d, const uint64_t& node) { return d + std::distance(fgt.edge_begin(node), fgt.edge_end(node)); });
    
    //Allocate Edges
    Edges.resize(num_out);
    EdgeImplTy* cur = &Edges[0];
    
    //Allocate In_Edges
    In_Edges.resize(num_in);
    EdgeImplTy* cur_In = &In_Edges[0]; //cur edgelist for in_edges
    
    //allocate Nodes
    Nodes.reserve(num);
    //std::cout << " I am : " <<Runtime::NetworkInterface::ID << " with Nodes = " << std::distance(p.first, p.second)<<"\n";

    //get ready to communicate
    Starts[Runtime::NetworkInterface::ID].first = NodePtr(&Nodes[0]);
    Starts[Runtime::NetworkInterface::ID].second = 2;
    
    //create nodes
    auto ii = p.first;
    auto iit = pt.first;
    for (; ii != p.second; ++ii, ++iit) {
      auto num_o = std::distance(fg.edge_begin(*ii), fg.edge_end(*ii));
      auto num_i = std::distance(fgt.edge_begin(*iit), fgt.edge_end(*iit));
      Nodes.emplace_back(cur, num_o, cur_In, num_i);      
      cur += num_o;
      cur_In += num_i;
      for (auto fgi = fg.edge_begin(*ii), fge = fg.edge_end(*ii);
           fgi != fge; ++fgi) {
        //        std::cout << *ii << " " << *fgi << " " << num_o << " " << num_i << " " << num << " " << num_out << " " << num_in << "\n";
        if (*fgi < num) { 
          addEdge(*ii, *fgi, Galois::MethodFlag::NONE);
        } else {
          std::cerr << "error in graph, dropping edge\n";
        }
      }
      for (auto fgti = fgt.edge_begin(*iit), fgte = fgt.edge_end(*iit);
           fgti != fgte; ++fgti)
        if (*fgti < num) {
          addInEdge(*iit, *fgti, Galois::MethodFlag::NONE);
        } else {
          std::cerr << "error in graph, dropping edge\n";
        }
    }
  }
  
  
  ~LC_Dist_InOut() {
    auto ii = iterator(this, Runtime::NetworkInterface::ID == 0 ? 0 : PrefixNum[Runtime::NetworkInterface::ID - 1] );
    auto ee = iterator(this, PrefixNum[Runtime::NetworkInterface::ID] );
    for (; ii != ee; ++ii)
      acquireNode(*ii, MethodFlag::ALL);
  }
  
  static void getStart(Runtime::PerHost<LC_Dist_InOut> graph, uint32_t whom) {
    //std::cerr << Runtime::NetworkInterface::ID << " getStart " << whom << "\n";
    Runtime::getSystemNetworkInterface().sendAlt(whom, putStart, graph, Runtime::NetworkInterface::ID, graph->Starts[Runtime::NetworkInterface::ID].first);
  }

  static void putStart(Runtime::PerHost<LC_Dist_InOut> graph, uint32_t whom, NodePtr start) {
    graph->Starts[whom].first = start;
    graph->Starts[whom].second = 2;
  }

  //blocking
  void fillStarts(uint32_t host) {
    if (Starts[host].second != 2) {
      int ex = 0;
      unsigned id = Runtime::NetworkInterface::ID;
      bool swap = Starts[host].second.compare_exchange_strong(ex,1);
      if (swap)
        Runtime::getSystemNetworkInterface().sendAlt(host, getStart, self, Runtime::NetworkInterface::ID);
      while (Starts[host].second != 2) { Runtime::doNetworkWork(); }
    }
    assert(Starts[host].first);
  }

  NodePtr makeNodePtr(uint64_t x) {
    auto ii = std::upper_bound(PrefixNum.begin(), PrefixNum.end(), x);
    assert(ii != PrefixNum.end());
    unsigned host = std::distance(PrefixNum.begin(), ii);
    unsigned offset = x - (host == 0 ? 0 : *(ii - 1));
    fillStarts(host);
    return Starts[host].first + offset;
  }

  void acquireNode(NodePtr node, Galois::MethodFlag mflag) {
    acquire(node, mflag);
  }

public:
  typedef EdgeImplTy edge_data_type;
  typedef NodeImplTy node_data_type;
  typedef typename EdgeData::reference edge_data_reference;
  typedef typename NodeData::reference node_data_reference;
  
  typedef boost::counting_iterator<uint64_t> iterator;
  typedef iterator local_iterator;
  typedef EdgeImplTy* edge_iterator;

  //! Creation and destruction
  typedef Runtime::PerHost<LC_Dist_InOut> pointer;
  static pointer allocate(std::vector<unsigned>& edges, std::vector<unsigned>& In_edges) {
    return pointer::allocate(edges, In_edges);
  }
  static pointer allocate(std::string& file, std::string& fileTr) {
    return pointer::allocate(file, fileTr);
  }
  static void deallocate(pointer ptr) {
    pointer::deallocate(ptr);
  }

  //! Mutation

  template<typename... Args>
  edge_iterator  addEdge(GraphNode src, GraphNode dst, MethodFlag mflag = MethodFlag::ALL) {
    NodePtr srcp = makeNodePtr(src);
    acquire(srcp, mflag);
    return srcp->append(dst);
  }

  // Adding inEdges to the Graph.
  template<typename... Args>
  edge_iterator addInEdge(GraphNode src, GraphNode dst, MethodFlag mflag = MethodFlag::ALL) {
    NodePtr srcp = makeNodePtr(src);
    acquire(srcp, mflag);
    return srcp->append_inEdges(dst);
  }
  //! Access

  NodeTy& at(GraphNode N, MethodFlag mflag = MethodFlag::ALL) {
    NodePtr np = makeNodePtr(N);
    acquire(np, mflag);
    return np->data;
  }

  unsigned get_num_outEdges(GraphNode N)
  {
    NodePtr np = makeNodePtr(N);
    return np->get_num_outEdges();
  }

  uint32_t getHost(GraphNode N) {
    return ((Galois::Runtime::fatPointer)makeNodePtr(N)).getHost();
  }

  //EdgeTy& at(edge_iterator E, MethodFlag mflag = MethodFlag::ALL) {
  //return E->data;
  //}

  GraphNode dst(edge_iterator E, MethodFlag mflag = MethodFlag::ALL) {
    return E->dst;
  }

  //! Capacity
  unsigned size() {
    return *PrefixNum.rbegin();
  }

  //! Iterators

  iterator begin() { return iterator(0); }
  iterator end  () { return iterator(PrefixNum[Runtime::NetworkInterface::Num - 1]); }

  local_iterator local_begin() {
    if (Runtime::LL::getTID() == 0)
      return iterator(Runtime::NetworkInterface::ID == 0 ? 0 : PrefixNum[Runtime::NetworkInterface::ID - 1] );
    else
      return local_end();
  }
  local_iterator local_end  () { return iterator(PrefixNum[Runtime::NetworkInterface::ID]); }

  edge_iterator edge_begin(GraphNode N, MethodFlag mflag = MethodFlag::ALL) {
    NodePtr np = makeNodePtr(N);
    acquire(np, mflag);
    if (mflag != MethodFlag::SRC_ONLY && mflag != MethodFlag::NONE)
      {
        /** prefetch **/
        for (auto ii = np->begin(), ee = np->end(); ii != ee; ++ii) {
          prefetch(makeNodePtr(ii->dst));
        }
        for (edge_iterator ii = np->begin(), ee = np->end(); ii != ee; ++ii) {
          acquireNode(makeNodePtr(ii->dst), mflag);
        }
      }
    return np->begin();
  }

  edge_iterator edge_end(GraphNode N, MethodFlag mflag = MethodFlag::ALL) {
    NodePtr np = makeNodePtr(N);
    acquireNode(np, mflag);
    return np->end();
  }


  /**
   * In Edge iterators 
   *
   */

  edge_iterator in_edge_begin(GraphNode N, MethodFlag mflag = MethodFlag::ALL) {
    NodePtr np = makeNodePtr(N);
    acquire(np, mflag);
    if (mflag != MethodFlag::SRC_ONLY && mflag != MethodFlag::NONE)
      {
        /** prefetch **/
        for (auto ii = np->begin_inEdges(), ee = np->end_inEdges(); ii != ee; ++ii) {
          prefetch(makeNodePtr(ii->dst));
        }
        for (edge_iterator ii = np->begin_inEdges(), ee = np->end_inEdges(); ii !=ee; ++ii) {
          acquireNode(makeNodePtr(ii->dst), mflag);	
        }
      }
    return np->begin_inEdges(); 
  }

  edge_iterator in_edge_end(GraphNode N, MethodFlag mflag = MethodFlag::ALL) {
    NodePtr np = makeNodePtr(N);
    acquireNode(np, mflag);
    return np->end_inEdges();  
  }

  template<typename Compare>
  void sort_edges(GraphNode N, Compare comp, MethodFlag mflag = MethodFlag::ALL) {
    std::sort(edge_begin(N, mflag), edge_end(N, mflag),
              [&comp] (const EdgeImplTy& e1, const EdgeImplTy& e2) {
                return comp(e1.dst, e1.data, e2.dst, e2.data);
              });
  }

  /*** Additional functions ***/
  void prefetch_all()
  {
    for(auto ii = this->begin(); ii != this->end(); ++ii)
      {
        prefetch(makeNodePtr(*ii));
      }
  }

  void prefetch_these(unsigned start_index, unsigned end_index)
  {
    auto ii = this->begin() +  start_index;
    auto ei = this->begin() + end_index;
    for(; ii != ei; ++ii)
      {
        prefetch(makeNodePtr(*ii));
      }
  }

  bool isLocal(GraphNode N) {
    NodePtr np = makeNodePtr(N);
    return np.isLocal();
  }

};

} //namespace Graph
} //namespace Galois

#endif
