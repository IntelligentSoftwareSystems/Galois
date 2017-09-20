/** Local computation graphs with in and out edges -*- C++ -*-
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
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 */
#ifndef LC_PARTITIONED_GRAPH_H
#define LC_PARTITIONED_GRAPH_H

#include "Galois/Graph/Details.h"

#include <boost/iterator/iterator_facade.hpp>
#include <boost/fusion/include/vector.hpp>
#include <boost/fusion/include/at_c.hpp>

#include <sstream>
#include <string>
#include <regex>

namespace galois {
namespace Graph {

struct read_lc_partitioned_graph_tag { };

/**
 * XXX update documentation
 * Modify a LC_Graph to have in and out edges. In edges are stored by value, so
 * modifying them does not modify the corresponding out edge.
 */
template<typename GraphTy>
class LC_Partitioned_Graph: public GraphTy::template with_id<true> {
  template<typename G>
  friend void readGraphDispatch(G&, read_lc_partitioned_graph_tag, const std::string&);

  typedef typename GraphTy::template with_id<true> MainGraph;
  typedef typename GraphTy::template with_id<true>::template with_node_data<void>::template with_no_lockable<true> PartGraph;

  template<typename G>
  class Part {
    G graph;
    std::pair<uint32_t,uint32_t> inRange;
    std::pair<uint32_t,uint32_t> outRange;
  }

  // XXX replicate node data?
  Part<MainGraph> mainPart;
  std::deque<Part<PartGraph>> parts;

  typename InGraph::GraphNode inGraphNode(typename Super::GraphNode n) {
    return inGraph.getNode(idFromNode(n));
  }

  void setNumParts(int i) {
    parts.resize(i - 1);
  }

  typedef MainGraph main_graph_type;
  typedef PartGraph part_graph_type;

public:
  typedef uint32_t GraphNode;
  typedef typename MainGraph::edge_data_type edge_data_type;
  typedef typename MainGraph::file_edge_data_type file_edge_data_type;
  typedef typename MainGraph::node_data_type node_data_type;
  typedef typename MainGraph::edge_data_reference edge_data_reference;
  typedef typename MainGraph::node_data_reference node_data_reference;
  typedef typename Super::edge_iterator edge_iterator;
  typedef typename Super::iterator iterator; // XXX Two level
  typedef typename Super::const_iterator const_iterator;
  typedef typename Super::local_iterator local_iterator;
  typedef typename Super::const_local_iterator const_local_iterator;
  typedef read_lc_partitioned_graph_tag read_tag;

  // Union of edge_iterator and InGraph::edge_iterator
  class in_edge_iterator: public boost::iterator_facade<in_edge_iterator, void*, boost::forward_traversal_tag, void*> {
    friend class boost::iterator_core_access;
    friend class LC_InOut_Graph;
    typedef edge_iterator Iterator0;
    typedef typename InGraph::edge_iterator Iterator1;
    typedef boost::fusion::vector<Iterator0, Iterator1> Iterators;

    Iterators its;
    LC_InOut_Graph* self;
    int type;

    void increment() { 
      if (type == 0)
        ++boost::fusion::at_c<0>(its);
      else
        ++boost::fusion::at_c<1>(its);
    }

    bool equal(const in_edge_iterator& o) const { 
      if (type != o.type)
        return false;
      if (type == 0) {
        return boost::fusion::at_c<0>(its) == boost::fusion::at_c<0>(o.its);
      } else {
        return boost::fusion::at_c<1>(its) == boost::fusion::at_c<1>(o.its);
      }
    }

    void* dereference() const { return 0; }

  public:
    in_edge_iterator(): type(0) { }
    in_edge_iterator(Iterator0 it): type(0) { boost::fusion::at_c<0>(its) = it; }
    in_edge_iterator(Iterator1 it, int): type(1) { boost::fusion::at_c<1>(its) = it; }
  };
  
  LC_InOut_Graph(): asymmetric(false) { }

  edge_data_reference getInEdgeData(in_edge_iterator ni, MethodFlag mflag = MethodFlag::UNPROTECTED) { 
    // galois::runtime::checkWrite(mflag, false);
    if (ni.type == 0) {
      return this->getEdgeData(boost::fusion::at_c<0>(ni.its));
    } else {
      return inGraph.getEdgeData(boost::fusion::at_c<1>(ni.its));
    }
  }

  GraphNode getInEdgeDst(in_edge_iterator ni) {
    if (ni.type == 0) {
      return this->getEdgeDst(boost::fusion::at_c<0>(ni.its));
    } else {
      return nodeFromId(inGraph.getId(inGraph.getEdgeDst(boost::fusion::at_c<1>(ni.its))));
    }
  }

  in_edge_iterator in_edge_begin(GraphNode N, MethodFlag mflag = MethodFlag::WRITE) {
    this->acquireNode(N, mflag);
    if (!asymmetric) {
      if (galois::runtime::shouldLock(mflag)) {
        for (edge_iterator ii = this->raw_begin(N), ei = this->raw_end(N); ii != ei; ++ii) {
          this->acquireNode(this->getEdgeDst(ii), mflag);
        }
      }
      return in_edge_iterator(this->raw_begin(N));
    } else {
      if (galois::runtime::shouldLock(mflag)) {
        for (typename InGraph::edge_iterator ii = inGraph.raw_begin(inGraphNode(N)),
            ei = inGraph.raw_end(inGraphNode(N)); ii != ei; ++ii) {
          this->acquireNode(nodeFromId(inGraph.getId(inGraph.getEdgeDst(ii))), mflag);
        }
      }
      return in_edge_iterator(inGraph.raw_begin(inGraphNode(N)), 0);
    }
  }

  in_edge_iterator in_edge_end(GraphNode N, MethodFlag mflag = MethodFlag::WRITE) {
    this->acquireNode(N, mflag);
    if (!asymmetric) {
      return in_edge_iterator(this->raw_end(N));
    } else {
      return in_edge_iterator(inGraph.raw_end(inGraphNode(N)), 0);
    }
  }

  detail::InEdgesIterator<LC_InOut_Graph> in_edges(GraphNode N, MethodFlag mflag = MethodFlag::WRITE) {
    return detail::InEdgesIterator<LC_InOut_Graph>(*this, N, mflag);
  }

  size_t idFromNode(GraphNode N) {
    return this->getId(N);
  }

  GraphNode nodeFromId(size_t N) {
    return this->getNode(N);
  }
};

template<typename GraphTy>
void readGraphDispatch(GraphTy& graph, read_lc_partitioned_graph_tag, const std::string& filename) {
  // partitioned files are in form xxx.gr.i.of.n
  std::regex parser("(.*?\\.)\\d+\\.of\\.(\\d+)\\Z", std::regex_constants::ECMAScript);
  std::smatch match;
  int parts = 0;
  std::string base;
  if (std::regex_match(filename, match, parser)) {
    base = match[1].str();
    std::ssub_match sub = match[2];
    std::istringstream(sub.str()) >> parts;
  } 
  
  if (parts == 0) {
    GALOIS_DIE("unknown name for partitioned graph");
  }

  graph.setNumParts(parts);
  for (int i = 0; i < parts; ++i) {
    std::ostringstream os(base);
    os << i << ".of." << parts;
    if (i == 0) {
      readGraphDispatch(graph.mainPart.graph, typename GraphTy::main_graph_type::read_tag, os.str());
    } else {
      readGraphDispatch(graph.part[i-1].graph, typename GraphTy::part_graph_type::read_tag, os.str());
    }
  }
}

}
}
#endif
