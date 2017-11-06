#ifndef APPS_BFS_HYBRIDBFS_H
#define APPS_BFS_HYBRIDBFS_H

#include "galois/Galois.h"
#include "galois/graphs/LCGraph.h"

template<typename NodeData,typename Dist>
struct HybridBFS {
  typedef typename galois::graphs::LC_CSR_Graph<NodeData,void>
    ::template with_no_lockable<true>::type
    ::template with_numa_alloc<true>::type 
    InnerGraph;
  typedef typename galois::graphs::LC_InOut_Graph<InnerGraph> Graph;
  typedef typename Graph::GraphNode GNode;

  typedef std::pair<GNode,Dist> WorkItem;
  typedef galois::InsertBag<GNode> NodeBag;
  typedef galois::InsertBag<WorkItem> WorkItemBag;
  
  galois::GAccumulator<size_t> count;
  NodeBag bags[2];

  template <typename I>
  void bfsPushBulkSyncOperator (I ii, Graph& graph, NodeBag* nextBag, Dist newDist) {
    GNode dst = graph.getEdgeDst(ii);
    NodeData& ddata = graph.getData(dst, galois::MethodFlag::UNPROTECTED);

    Dist oldDist;
    while (true) {
      oldDist = ddata.dist;
      if (oldDist <= newDist)
        return;
      if (__sync_bool_compare_and_swap(&ddata.dist, oldDist, newDist)) {
        nextBag->push(dst);
        this->count += 1 
          + std::distance(graph.edge_begin(dst, galois::MethodFlag::UNPROTECTED),
              graph.edge_end(dst, galois::MethodFlag::UNPROTECTED));
        break;
      }
    }
  }

  void bfsPushAsync(Graph& graph, WorkItemBag& asyncBag) {

    using namespace galois::worklists;
    typedef dChunkedLIFO<256> WL;
    typedef BulkSynchronous<dChunkedLIFO<256> > BSWL;

    galois::for_each(galois::iterate(asyncBag), 
        [&] (const WorkItem& item, auto& ctx) {
          GNode n = item.first;
          const Dist& newDist = item.second;

          for (typename Graph::edge_iterator ii = graph.edge_begin(n, galois::MethodFlag::UNPROTECTED),
                ei = graph.edge_end(n, galois::MethodFlag::UNPROTECTED); ii != ei; ++ii) {

            GNode dst = graph.getEdgeDst(ii);
            NodeData& ddata = graph.getData(dst, galois::MethodFlag::UNPROTECTED);

            Dist oldDist;
            while (true) {
              oldDist = ddata.dist;
              if (oldDist <= newDist)
                return;
              if (__sync_bool_compare_and_swap(&ddata.dist, oldDist, newDist)) {
                ctx.push(WorkItem(dst, newDist + 1));
                break;
              }
            }
          }
        },
        galois::no_conflicts(),
        galois::loopname("bfsPushAsync"),
        galois::wl<BSWL>());
  }

  void bfsPullTopo (Graph& graph, NodeBag* nextBag, const Dist& newDist) {

    galois::do_all(galois::iterate(graph),
        [&, outer=this] (const GNode& n) {
          NodeData& sdata = graph.getData(n, galois::MethodFlag::UNPROTECTED);
          if (sdata.dist <= newDist)
            return;

          for (auto ii = graph.in_edge_begin(n, galois::MethodFlag::UNPROTECTED),
                ei = graph.in_edge_end(n, galois::MethodFlag::UNPROTECTED); ii != ei; ++ii) {

            GNode dst = graph.getInEdgeDst(ii);
            NodeData& ddata = graph.getData(dst, galois::MethodFlag::UNPROTECTED);

            if (ddata.dist + 1 == newDist) {
              sdata.dist = newDist;
              nextBag->push(n);
              outer->count += 1
                + std::distance(graph.edge_begin(n, galois::MethodFlag::UNPROTECTED),
                  graph.edge_end(n, galois::MethodFlag::UNPROTECTED));
              break;
            }
          }

        },
        galois::loopname("bfsPullTopo"));
  }


  void operator()(Graph& graph, const GNode& source) {

    int next = 0;
    Dist newDist = 1;
    int numForward = 0;
    int numBackward = 0;

    graph.getData(source).dist = 0;

    if (std::distance(graph.edge_begin(source), graph.edge_end(source)) + 1 > (long) graph.sizeEdges() / 20) {
      bfsPullTopo(graph, &bags[next], newDist);
      numBackward += 1;

    } else {
      galois::do_all(
          galois::iterate(graph.out_edges(source, galois::MethodFlag::UNPROTECTED).begin()
            , graph.out_edges(source, galois::MethodFlag::UNPROTECTED).end()),

        [&, outer=this] (auto ii) {
          outer->bfsPushBulkSyncOperator(ii, graph, &outer->bags[next], newDist);
        },
        galois::loopname("bfsPushBulkSync"));


      numForward += 1;
    }

    while (!bags[next].empty()) {
      size_t nextSize = count.reduce();
      count.reset();
      int cur = next;
      next = (cur + 1) & 1;
      newDist++;

      if (nextSize > graph.sizeEdges() / 20) {
        //std::cout << "Dense " << nextSize << "\n";
        bfsPullTopo(graph, &bags[next], newDist);
        numBackward += 1;

      } else if (numForward < 10 && numBackward == 0) {
        //std::cout << "Sparse " << nextSize << "\n";
        
        galois::do_all(galois::iterate(bags[cur]),
            [&, outer=this] (const GNode& n) {
              for (typename Graph::edge_iterator ii = graph.edge_begin(n, galois::MethodFlag::UNPROTECTED),
                    ei = graph.edge_end(n, galois::MethodFlag::UNPROTECTED); ii != ei; ++ii) {
                outer->bfsPushBulkSyncOperator(ii, graph, &outer->bags[next], newDist);
              }
            },
            galois::loopname("bfsPushBulkSync"));
        numForward += 1;
      } else {
        //std::cout << "Async " << nextSize << "\n";
        WorkItemBag asyncBag;

        galois::do_all(galois::iterate(bags[cur]), 
            [&] (const GNode& n) {
              asyncBag.push(WorkItem(n, newDist));
            });

        bfsPushAsync(graph, asyncBag);
        break;
      }

      bags[cur].clear();
    }
  }
};

#endif
