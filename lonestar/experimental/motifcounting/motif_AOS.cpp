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

#include "galois/Galois.h"
#include "galois/Reduction.h"
#include "galois/Bag.h"
#include "galois/Timer.h"
#include "galois/graphs/LCGraph.h"
#include "galois/ParallelSTL.h"
#include "llvm/Support/CommandLine.h"
#include "Lonestar/BoilerPlate.h"
#include "galois/gstl.h"

#include "galois/runtime/Profile.h"

#include <boost/iterator/transform_iterator.hpp>

#include <utility>
#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>

const char* name = "k-motif";
const char* desc = "Counts the k-motifs in a graph";
const char* url  = 0;

enum Algo {
  nodeiteratorpre,
  nodeiteratorpre_moreCompute
};

namespace cll = llvm::cl;
static cll::opt<std::string>
    inputFilename(cll::Positional, cll::desc("<input file>"), cll::Required);
static cll::opt<Algo> algo(
    "algo", cll::desc("Choose an algorithm:"),
    cll::values(
      clEnumValN(Algo::nodeiteratorpre_moreCompute, "nodeiteratorpre_moreCompute", "Node Iterator with pushing singleton and more compute"),
      clEnumValN(Algo::nodeiteratorpre, "nodeiteratorpre", "Node Iterator (default)"),
                clEnumValEnd),
    cll::init(Algo::nodeiteratorpre));
static cll::opt<unsigned int>
    k("k",
        cll::desc("max number of vertices in k-motif(default value 0)"),
        cll::init(0));


struct NodeData{
  uint64_t ea;
  uint64_t bb;
};


//typedef galois::graphs::LC_CSR_Graph<NodeData, void>::with_numa_alloc<
typedef galois::graphs::LC_CSR_Graph<uint8_t, void>::with_numa_alloc<
true>::type ::with_no_lockable<true>::type Graph;
// typedef galois::graphs::LC_CSR_Graph<uint32_t,void> Graph;
// typedef galois::graphs::LC_Linear_Graph<uint32_t,void> Graph;

typedef Graph::GraphNode GNode;

/**
 * Like std::lower_bound but doesn't dereference iterators. Returns the first
 * element for which comp is not true.
 */
template <typename Iterator, typename Compare>
Iterator lowerBound(Iterator first, Iterator last, Compare comp) {
  Iterator it;
  typename std::iterator_traits<Iterator>::difference_type count, half;
  count = std::distance(first, last);
  while (count > 0) {
    it   = first;
    half = count / 2;
    std::advance(it, half);
    if (comp(it)) {
      first = ++it;
      count -= half + 1;
    } else {
      count = half;
    }
  }
  return first;
}

/**
 * Like std::upper_bound but doesn't dereference iterators. Returns the first
 * element for which comp is not true.
 */
template <typename Iterator, typename Compare>
Iterator upperBound(Iterator first, Iterator last, Compare comp) {
  Iterator it;
  typename std::iterator_traits<Iterator>::difference_type count, half;
  count = std::distance(first, last);
  while (count > 0) {
    it   = first;
    half = count / 2;
    std::advance(it, half);
    if (!comp(it)) {
      first = ++it;
      count -= half + 1;
    } else {
      count = half;
    }
  }
  return first;
}

/**
 * std::set_intersection over edge_iterators.
 */
template <typename G>
size_t countEqual(G& g, typename G::edge_iterator aa,
                  typename G::edge_iterator ea, typename G::edge_iterator bb,
                  typename G::edge_iterator eb) {
  size_t retval = 0;
  while (aa != ea && bb != eb) {
    typename G::GraphNode a = g.getEdgeDst(aa);
    typename G::GraphNode b = g.getEdgeDst(bb);
    if (a < b) {
      ++aa;
    } else if (b < a) {
      ++bb;
    } else {
      retval += 1;
      ++aa;
      ++bb;
    }
  }
  return retval;
}

template <typename G>
struct LessThan {
  G& g;
  typename G::GraphNode n;
  LessThan(G& g, typename G::GraphNode n) : g(g), n(n) {}
  bool operator()(typename G::edge_iterator it) { return g.getEdgeDst(it) < n; }
};
template <typename G>
struct LessThanOrEqual {
  G& g;
  typename G::GraphNode n;
  LessThanOrEqual(G& g, typename G::GraphNode n) : g(g), n(n) {}
  bool operator()(typename G::edge_iterator it) { return g.getEdgeDst(it) <= n; }
};


template <typename G>
struct GreaterThanOrEqual {
  G& g;
  typename G::GraphNode n;
  GreaterThanOrEqual(G& g, typename G::GraphNode n) : g(g), n(n) {}
  bool operator()(typename G::edge_iterator it) {
    return !(n < g.getEdgeDst(it));
  }
};

template <typename G>
struct GreaterThan{
  G& g;
  typename G::GraphNode n;
  GreaterThan(G& g, typename G::GraphNode n) : g(g), n(n) {}
  bool operator()(typename G::edge_iterator it) {
    return (n > g.getEdgeDst(it));
  }
};




template <typename G>
struct DegreeLess : public std::binary_function<typename G::GraphNode,
                                                typename G::GraphNode, bool> {
  typedef typename G::GraphNode N;
  G* g;
  DegreeLess(G& g) : g(&g) {}

  bool operator()(const N& n1, const N& n2) const {
    return std::distance(g->edge_begin(n1), g->edge_end(n1)) <
           std::distance(g->edge_begin(n2), g->edge_end(n2));
  }
};

template <typename G>
struct GetDegree
    : public std::unary_function<typename G::GraphNode, ptrdiff_t> {
  typedef typename G::GraphNode N;
  G* g;
  GetDegree(G& g) : g(&g) {}

  ptrdiff_t operator()(const N& n) const {
    return std::distance(g->edge_begin(n), g->edge_end(n));
  }
};

template <typename GraphNode, typename EdgeTy>
struct IdLess {
  bool
  operator()(const galois::graphs::EdgeSortValue<GraphNode, EdgeTy>& e1,
             const galois::graphs::EdgeSortValue<GraphNode, EdgeTy>& e2) const {
    return e1.dst < e2.dst;
  }
};

template <typename VecTy, typename ElemTy>
bool vertexNotInTuple(const VecTy& vec, const ElemTy& elem){
  return (std::find(vec.begin(), vec.end(), elem) == vec.end());
}
template <typename VecTy, typename ElemTy, typename KeyTy>
bool isPresentOnLeft(const VecTy& vec, const ElemTy& elem, const KeyTy& key){
  return (std::find(vec.begin(), vec.begin() + key, elem) != (vec.begin() + key));
}
template <typename VecTy>
size_t uniqueInTuple(const VecTy& vec){
  return (std::set<GNode>(vec.begin(), vec.end()).size());
}
template <typename VecTy, typename ElemTy, typename StInfoTy, typename KeyTy>
bool edgeNotInTuple(const VecTy& vec, const ElemTy& elem, const StInfoTy& st_info, const KeyTy& key){
  auto it = std::find(vec.begin(), vec.end(), elem); 
  if (it == vec.end()) {
    return true;
  } else {
    auto index = std::distance(vec.begin(), it);
    return !(st_info[index] == key);
  }
}

typedef galois::gstl::Vector<GNode> VecGNodeTy;
typedef galois::gstl::Vector<uint8_t> VecUnsignedTy;
void nodeIteratingAlgoWithStruct(Graph& graph) {

  struct SubGraphTuple {
    VecGNodeTy vertices;
    uint8_t key;
    VecUnsignedTy st_info;
    SubGraphTuple(const VecGNodeTy& v1, const uint8_t& k1, const VecUnsignedTy& s1) : vertices(v1), key(k1), st_info(s1) {}
  };
  galois::InsertBag<SubGraphTuple> items;
  galois::InsertBag<SubGraphTuple> items_active;
  galois::InsertBag<SubGraphTuple> items_final;
  galois::GAccumulator<size_t> kMotifCount;
  galois::GAccumulator<size_t> numClosedStructures;

       galois::do_all(
            galois::iterate(graph),
            [&](const GNode& n) {
              auto& ndata = graph.getData(n);
              Graph::edge_iterator first =
                  graph.edge_begin(n, galois::MethodFlag::UNPROTECTED);
              Graph::edge_iterator last =
                  graph.edge_end(n, galois::MethodFlag::UNPROTECTED);
              //Graph::edge_iterator ea = first + ndata.ea; // std::advance(first, ndata.ea);
              //Graph::edge_iterator bb = first + ndata.bb; //std::advance(first, ndata.bb);

              Graph::edge_iterator ea =
                  lowerBound(first, last, LessThan<Graph>(graph, n));
              Graph::edge_iterator bb =
                  lowerBound(first, last, GreaterThanOrEqual<Graph>(graph, n));

              for (; bb != last; ++bb) {
                GNode B = graph.getEdgeDst(bb);
                items_active.push(SubGraphTuple(VecGNodeTy{n,B}, 0, VecUnsignedTy{1,0}));
                items_active.push(SubGraphTuple(VecGNodeTy{n,B}, 1, VecUnsignedTy{1,0}));
              }
            },
            galois::chunk_size<512>(), galois::steal(),
            galois::loopname("nodeIteratingAlgoWithStruct"));

       //Optimization
       //1. exclude tuples that cannot be expanded: Early pruning

    std::cout << "Start phase 2\n";
    if (k > 2){
      //galois::do_all(
      galois::for_each(
            galois::iterate(items_active),
            //[&](const SubGraphTuple& sg) {
            [&](const SubGraphTuple& sg, auto& ctx) {
              auto n = sg.vertices[sg.key];
              Graph::edge_iterator first =
                  graph.edge_begin(n, galois::MethodFlag::UNPROTECTED);
              Graph::edge_iterator last =
                  graph.edge_end(n, galois::MethodFlag::UNPROTECTED);

              //auto first_elem = sg.vertices[0];
              auto max_elem = sg.vertices[0];
              /*
               * If n is duplicated, find the first instance of n else it will find n:
               * n is gauranteed to be present in the tuple
               */
              auto first_instance_of_n = std::distance(sg.vertices.begin(), std::find(sg.vertices.begin(), sg.vertices.end(), n));
              /*
               * Find the largest element on the right side of the key
               * If no duplicates are present than first_instance_of_n == sg.key
               */
              //auto first_instance_of_n  = std::distance(sg.vertices.begin(), first_instance_of_n_it);
              if(sg.vertices.begin() + first_instance_of_n + 1 < sg.vertices.end()){
                auto max_elem_after_n  = std::max_element(sg.vertices.begin() + first_instance_of_n + 1, sg.vertices.end());
                if (max_elem_after_n != sg.vertices.end()) {
                  max_elem = std::max(*max_elem_after_n, max_elem);
                }
              }

              Graph::edge_iterator bb = lowerBound(
                  first, last, LessThan<Graph>(graph, max_elem));

              for (; bb != last; ++bb) {
                GNode dst = graph.getEdgeDst(bb);
                auto verts = sg.vertices;
                auto st_info = sg.st_info;
                if (isPresentOnLeft(verts, dst, sg.key)) {
                    continue;
                }
                //Do not add duplicate edges
                if (edgeNotInTuple(verts, dst, st_info, sg.key)){
                  /**
                   * If adding duplicate nodes, check the order of structure
                   */
                  auto tupleSize = verts.size();
                  if (verts[tupleSize - 1] == dst) {
                    /*
                     * Must be increasing order. Should never be equal, since edgeNotInTuple should discard it
                     */
                    if (st_info[tupleSize - 1] >= sg.key)
                      continue;
                  }
                  verts.push_back(dst);
                  st_info.push_back(sg.key);
                  //Should have k unique elements
                  if (uniqueInTuple(verts) == k) {
                    items_final.push(SubGraphTuple(verts, sg.key, st_info));
                    kMotifCount += 1;

#if 0
                    /**
                     * Specializing for finding the closed structures for the 3-motif
                     * Only need to look at the end points
                     *
                     * Find close structures using same rules of canonicality
                     * Only look at the last nodes only
                     */

                    if (k == 3) {
                      if(st_info[st_info.size() - 1] == st_info.size() - 2){
                        auto n_last = verts[st_info.size()-1];

                        if(n_last > verts[st_info.size() - 2]){
                          Graph::edge_iterator first_local =
                            graph.edge_begin(n_last, galois::MethodFlag::UNPROTECTED);
                          Graph::edge_iterator last_local =
                            graph.edge_end(n_last , galois::MethodFlag::UNPROTECTED);
                          Graph::edge_iterator bb = lowerBound(
                              first_local, last_local, LessThan<Graph>(graph, verts[0]));
                          auto dst = graph.getEdgeDst(bb);
                          if(dst == verts[0]){

                            verts.push_back(dst);
                            st_info.push_back(st_info.size() - 1);
                            items_final.push(SubGraphTuple(verts, st_info.size() - 1, st_info));
                            kMotifCount += 1;
                            numClosedStructures += 1;
                          }
                        }
                      }
                    } else {
#endif
                      std::set<GNode> local_set{verts.begin(), verts.end()};
                      for (auto v : local_set) {
                        auto first_instance_of_v_it = std::find(verts.begin(), verts.end(), v);
                        auto i = std::distance(verts.begin(), first_instance_of_v_it);
                        auto edge_support = std::count(st_info.begin() + 2, st_info.end(), i);
                        if (edge_support < k - 2) {
#if 0
                          Graph::edge_iterator first_v =
                            graph.edge_begin(v, galois::MethodFlag::UNPROTECTED);
                          Graph::edge_iterator last_v =
                            graph.edge_end(v, galois::MethodFlag::UNPROTECTED);
                          auto num_edges = std::distance(first_v, last_v);
                          /*
                           * Only add if there is a possibility of getting new structures
                           */
                          if(num_edges > k - (edge_support + 1)){
#endif
                            //items_active.push(SubGraphTuple(verts, i, st_info));
                            ctx.push(SubGraphTuple(verts, i, st_info));
                            numClosedStructures +=1;
                          //}
                        }
                      }
                    //}
                  }
                  else if (uniqueInTuple(verts) < k) {
                    //Only push unique elements
                    std::set<GNode> local_set{verts.begin(), verts.end()};
                    for (auto v : local_set) {
                      auto first_instance_of_v_it = std::find(verts.begin(), verts.end(), v);
                      auto i = std::distance(verts.begin(), first_instance_of_v_it);
                      //items_active.push(SubGraphTuple(verts, i, st_info));
                      ctx.push(SubGraphTuple(verts, i, st_info));
                    }
                  }
                }
              }
            },
            galois::chunk_size<128>(), galois::steal(), galois::no_conflicts(),
            galois::wl<galois::worklists::PerSocketChunkFIFO<128>>(),
            galois::loopname("nodeIteratingAlgoWithStruct"));
            } else {
                items_final.swap(items_active);
                items_active.clear();
            }

#if 0
  std::cout << "items2" << "\n";
  for(auto ii = items_final.begin(); ii != items_final.end(); ++ii){
    for(auto i :  (*ii).vertices){
      std::cout << i << "--";
    }
    std::cout << "key : " << (uint32_t)(*ii).key << "\n";
    std::cout <<"\n";
    for(auto i :  (*ii).st_info){
      std::cout <<  (uint32_t)i << "--";
    }
    std::cout << "\n\n";
  }
#endif

  //std::cout << "Num " << k << "-motif: " << std::distance(items_final.begin(), items_final.end()) << "\n";
  std::cout << "Num " << k << "-motif: " << kMotifCount.reduce() << "\n";
  std::cout << "NumClosedStructures: " << numClosedStructures.reduce() << "\n";
}


void kMotif_moreCompute(Graph& graph) {

  struct SubGraphTuple {
    VecGNodeTy vertices;
    VecUnsignedTy st_info;
    SubGraphTuple(const VecGNodeTy& v1, const VecUnsignedTy& s1) : vertices(v1), st_info(s1) {}
  };
  galois::InsertBag<SubGraphTuple> items;
  galois::InsertBag<SubGraphTuple> items_active;
  galois::InsertBag<SubGraphTuple> items_active2;
  galois::InsertBag<SubGraphTuple> items_final;
  galois::GAccumulator<size_t> kMotifCount;
  galois::GAccumulator<size_t> numClosedStructures;
  galois::GAccumulator<size_t> numPushItems_loops;
  galois::GAccumulator<size_t> numPushItems;
  galois::GAccumulator<size_t> itemSizeBytes;

       galois::do_all(
            galois::iterate(graph),
            [&](const GNode& n) {
              auto& ndata = graph.getData(n);
              Graph::edge_iterator first =
                  graph.edge_begin(n, galois::MethodFlag::UNPROTECTED);
              Graph::edge_iterator last =
                  graph.edge_end(n, galois::MethodFlag::UNPROTECTED);

              Graph::edge_iterator ea =
                  lowerBound(first, last, LessThan<Graph>(graph, n));
              Graph::edge_iterator bb =
                  lowerBound(first, last, GreaterThanOrEqual<Graph>(graph, n));

              for (; bb != last; ++bb) {
                GNode B = graph.getEdgeDst(bb);
                items_active.push(SubGraphTuple(VecGNodeTy{n,B}, VecUnsignedTy{1,0}));
              }
            },
            galois::chunk_size<512>(), galois::steal(),
            galois::loopname("nodeIteratingAlgoWithStruct"));

       //Optimization
       //1. exclude tuples that cannot be expanded: Early pruning

    std::cout << "Start phase 2\n";
    if (k > 2){
      while(std::distance(items_active.begin(), items_active.end()) > 0){
      galois::do_all(
      //galois::for_each(
            galois::iterate(items_active),
            [&](const SubGraphTuple& sg) {
            //[&](const SubGraphTuple& sg, auto& ctx) {

              std::set<GNode> local_set{sg.vertices.begin(), sg.vertices.end()};
              for (auto n : local_set) {
                auto first_instance_of_n_it = std::find(sg.vertices.begin(), sg.vertices.end(), n);
                auto key = std::distance(sg.vertices.begin(), first_instance_of_n_it);

              //auto n = sg.vertices[key];
              Graph::edge_iterator first =
                  graph.edge_begin(n, galois::MethodFlag::UNPROTECTED);
              Graph::edge_iterator last =
                  graph.edge_end(n, galois::MethodFlag::UNPROTECTED);

              //auto first_elem = sg.vertices[0];
              auto max_elem = sg.vertices[0];
              /*
               * If n is duplicated, find the first instance of n else it will find n:
               * n is gauranteed to be present in the tuple
               */
              //auto first_instance_of_n = std::distance(sg.vertices.begin(), std::find(sg.vertices.begin(), sg.vertices.end(), n));
              /*
               * Find the largest element on the right side of the key
               * If no duplicates are present than first_instance_of_n == sg.key
               */
              //auto first_instance_of_n  = std::distance(sg.vertices.begin(), first_instance_of_n_it);
              if(sg.vertices.begin() + key + 1 < sg.vertices.end()){
                auto max_elem_after_n  = std::max_element(sg.vertices.begin() + key + 1, sg.vertices.end());
                if (max_elem_after_n != sg.vertices.end()) {
                  max_elem = std::max(*max_elem_after_n, max_elem);
                }
              }

              Graph::edge_iterator bb = lowerBound(
                  first, last, LessThan<Graph>(graph, max_elem));

              for (; bb != last; ++bb) {
                GNode dst = graph.getEdgeDst(bb);
                auto verts = sg.vertices;
                auto st_info = sg.st_info;
                if (isPresentOnLeft(verts, dst, key)) {
                    continue;
                }
                //Do not add duplicate edges
                //if (edgeNotInTuple(verts, dst, st_info[key])){
                if (edgeNotInTuple(verts, dst, st_info, key)){
                  /**
                   * If adding duplicate nodes, check the order of structure
                   */
                  auto tupleSize = verts.size();
                  if (verts[tupleSize - 1] == dst) {
                    /*
                     * Must be increasing order. Should never be equal, since edgeNotInTuple should discard it
                     */
                    if (st_info[tupleSize - 1] >= key)
                      continue;
                  }
                  verts.push_back(dst);
                  st_info.push_back(key);
                  //Should have k unique elements
                  if (uniqueInTuple(verts) == k) {
                    //items_final.push(SubGraphTuple(verts, st_info));
                    kMotifCount += 1;

                    bool can_expand = false;
                    std::set<GNode> local_set{verts.begin(), verts.end()};
                    for (auto v : local_set) {
                      auto first_instance_of_v_it = std::find(verts.begin(), verts.end(), v);
                      auto i = std::distance(verts.begin(), first_instance_of_v_it);
                      auto edge_support = std::count(st_info.begin() + 2, st_info.end(), i);
                      if (edge_support < k - 2) {
                        can_expand = true;
                        break;
                      }
                    }

                    if (can_expand) {
                      numPushItems_loops += 1;
                      itemSizeBytes += (sizeof(SubGraphTuple) +  sizeof(VecGNodeTy) + sizeof(GNode)*verts.size() +  sizeof(VecUnsignedTy) + sizeof(uint8_t)*st_info.size());
                      items_active2.push(SubGraphTuple(verts, st_info));
                      //items_active.push(SubGraphTuple(verts, st_info));
                      //ctx.push(SubGraphTuple(verts, st_info));
                    }
                  }
                    else if (uniqueInTuple(verts) < k) {
                      numPushItems += 1;
                      items_active2.push(SubGraphTuple(verts, st_info));
                      //ctx.push(SubGraphTuple(verts, st_info));
                  }
                }
              }
            }
            },
            //galois::chunk_size<128>(), galois::steal(), galois::no_conflicts(),
            //galois::wl<galois::worklists::PerSocketChunkFIFO<128>>(),
            galois::loopname("nodeIteratingAlgoWithStruct"));

              std::cout << "Num Push items : "  << numPushItems.reduce() << "\n";
              std::cout << "Num Push items loop : " << numPushItems_loops.reduce() << "\n";
              std::cout << "items2" << "\n";

#if 0
              for(auto ii = items_active2.begin(); ii != items_active2.end(); ++ii){
                for(auto i :  (*ii).vertices){
                  std::cout << i << "--";
                }
                std::cout <<"\n";
                for(auto i :  (*ii).st_info){
                  std::cout <<  (uint32_t)i << "--";
                }
                std::cout << "\n\n";
              }
#endif 


              items_active.swap(items_active2);
              items_active2.clear();
            }
            } else {
                items_final.swap(items_active);
                items_active.clear();
            }

#if 0
  std::cout << "items2" << "\n";
  for(auto ii = items_final.begin(); ii != items_final.end(); ++ii){
    for(auto i :  (*ii).vertices){
      std::cout << i << "--";
    }
    std::cout <<"\n";
    for(auto i :  (*ii).st_info){
      std::cout <<  (uint32_t)i << "--";
    }
    std::cout << "\n\n";
  }
#endif

  //std::cout << "Num " << k << "-motif: " << std::distance(items_final.begin(), items_final.end()) << "\n";
  std::cout << "Num " << k << "-motif: " << kMotifCount.reduce() << "\n";
  std::cout << "NumClosedStructures: " << numClosedStructures.reduce() << "\n";
  std::cout << "Num Push items : "  << numPushItems.reduce() << "\n";
  std::cout << "Num Push items loop : " << numPushItems_loops.reduce() << "\n";
  std::cout << "Num Push items bytes : " << itemSizeBytes.reduce() << "\n";
  std::cout << "SubGraphTuple bytes : " << sizeof(SubGraphTuple) << "\n";
  std::cout << "GNode vec bytes : " << sizeof(VecGNodeTy) << "\n";
  std::cout << "Uint8 vec bytes : " << sizeof(VecUnsignedTy) << "\n";
}

void makeGraph(Graph& graph, const std::string& triangleFilename) {
  typedef galois::graphs::FileGraph G;
  typedef G::GraphNode N;

  G initial, permuted;

  initial.fromFileInterleaved<void>(inputFilename);

  // Getting around lack of resize for deque
  std::deque<N> nodes;
  std::copy(initial.begin(), initial.end(), std::back_inserter(nodes));
  // Sort by degree
  galois::ParallelSTL::sort(nodes.begin(), nodes.end(), DegreeLess<G>(initial));

  std::deque<N> p;
  std::copy(nodes.begin(), nodes.end(), std::back_inserter(p));
  // Transpose
  size_t idx = 0;
  for (N n : nodes) {
    p[n] = idx++;
  }

  galois::graphs::permute<void>(initial, p, permuted);
  galois::do_all(galois::iterate(permuted),
                 [&](N x) { permuted.sortEdges<void>(x, IdLess<N, void>()); });

  std::cout << "Writing new input file: " << triangleFilename << "\n";
  permuted.toFile(triangleFilename);
  galois::gPrint("loading file after creating triangleFilename\n");
  galois::graphs::readGraph(graph, permuted);
  //graph.allocateAndLoadGraph(triangleFilename);
}

void readGraph(Graph& graph) {
  if (inputFilename.find(".gr.triangles") !=
      inputFilename.size() - strlen(".gr.triangles")) {
    // Not directly passed .gr.triangles file
    std::string triangleFilename = inputFilename + ".triangles";
    std::ifstream triangleFile(triangleFilename.c_str());
    if (!triangleFile.good()) {
      // triangles doesn't already exist, create it
      galois::gPrint("Start makeGraph\n");
      makeGraph(graph, triangleFilename);
      galois::gPrint("Done makeGraph\n");
    } else {
      // triangles does exist, load it
      galois::gPrint("Start loading", triangleFilename, "\n");
      galois::graphs::readGraph(graph, triangleFilename);
      //graph.allocateAndLoadGraph(triangleFilename);
      galois::gPrint("Done loading", triangleFilename, "\n");
    }
  } else {
    galois::gPrint("Start loading", inputFilename, "\n");
    galois::graphs::readGraph(graph, inputFilename);
    //graph.allocateAndLoadGraph(inputFilename);
    galois::gPrint("Done loading", inputFilename, "\n");
  }

  size_t index = 0;
  for (GNode n : graph) {
    //graph.getData(n) = index++;
  }
}

int main(int argc, char** argv) {
  galois::SharedMemSys G;
  LonestarStart(argc, argv, name, desc, url);

  Graph graph;

  galois::StatTimer Tinitial("GraphReadingTime");
  galois::gPrint("Start readGraph\n");
  Tinitial.start();
  readGraph(graph);
  Tinitial.stop();
  galois::gPrint("Done readGraph\n");

  galois::preAlloc(600);
  //galois::preAlloc(numThreads + 16 * (graph.size() + graph.sizeEdges()) /
                                    //galois::runtime::pagePoolSize());
  galois::reportPageAlloc("MeminfoPre");

  galois::StatTimer T;
  T.start();
  // case by case preAlloc to avoid allocating unnecessarily
  switch (algo) {
  case nodeiteratorpre:
    //nodeIteratingAlgoPre(graph);
    nodeIteratingAlgoWithStruct(graph);
    break;
  case nodeiteratorpre_moreCompute:
    //nodeIteratingAlgoPre(graph);
    kMotif_moreCompute(graph);
    break;
  default:
    std::cerr << "Unknown algo: " << algo << "\n";
  }
  T.stop();

  galois::reportPageAlloc("MeminfoPost");
  return 0;
}
