#ifndef APPS_BFS_HYBRIDBFS_H
#define APPS_BFS_HYBRIDBFS_H

#include "galois/Galois.h"
#include "galois/graphs/LCGraph.h"

template<typename NodeData, typename DistType>
struct HybridBFS {
  using InnerGraph = typename galois::graphs::LC_CSR_Graph<NodeData,void>
                                ::template with_no_lockable<true>::type
                                ::template with_numa_alloc<true>::type;
  using Graph = typename galois::graphs::LC_InOut_Graph<InnerGraph>;
  using GNode = typename Graph::GraphNode;

  using WorkItem = std::pair<GNode, DistType>;
  using NodeBag = galois::InsertBag<GNode>;
  using WorkItemBag = galois::InsertBag<WorkItem>;

  // used to track how much work was done in a round to determine if you do a
  // push or a pull
  galois::GAccumulator<size_t> count; 
  // 2 bags; a "current" bag and a "new" bag to flip between BSP phases
  NodeBag bags[2]; 

  /**
   * Push operator. Takes an edge and does the update to the destination 
   * if the distance on the dest is higher than the distance we are using
   * to update.
   *
   * @tparam I type of an Edge
   *
   * @param outEdge edge to consider/push out of 
   * @param graph graph object
   * @param nextBag Nodes that are updated are added to this bag
   * @param newDist Distance to push along edge
   */
  template <typename I>
  void bfsPushBulkSyncOperator(I outEdge, Graph& graph, NodeBag* nextBag, 
                               DistType newDist) {
    GNode dst = graph.getEdgeDst(outEdge);
    NodeData& ddata = graph.getData(dst, galois::MethodFlag::UNPROTECTED);

    DistType oldDist;
    while (true) {
      oldDist = ddata.dist;

      if (oldDist <= newDist) return;

      if (__sync_bool_compare_and_swap(&ddata.dist, oldDist, newDist)) {
        nextBag->push(dst);
        this->count += 1 + 
          std::distance(graph.edge_begin(dst, galois::MethodFlag::UNPROTECTED),
                        graph.edge_end(dst, galois::MethodFlag::UNPROTECTED));
        break;
      }
    }
  }

  /**
   * Does BFS push, asynchronous, using a worklist. Finishes the rest of BFS
   * (i.e. not round by round).
   *
   * @param graph graph object to do BFS on
   * @param asyncBag Bag of nodes that need to be processed
   */
  void bfsPushAsync(Graph& graph, WorkItemBag& asyncBag) {
    using namespace galois::worklists;
    using BSWL = BulkSynchronous<dChunkedLIFO<256>>;

    // each thread processes one node + does pushes along its edges
    galois::for_each(
      galois::iterate(asyncBag), 
      [&] (const WorkItem& item, auto& ctx) {
        GNode n = item.first;
        const DistType& newDist = item.second;

        for (auto ii = graph.edge_begin(n, galois::MethodFlag::UNPROTECTED),
                  ei = graph.edge_end(n, galois::MethodFlag::UNPROTECTED); 
             ii != ei;
             ++ii) {
          GNode dst = graph.getEdgeDst(ii);
          NodeData& ddata = graph.getData(dst, galois::MethodFlag::UNPROTECTED);

          DistType oldDist;
          while (true) {
            oldDist = ddata.dist;
            if (oldDist <= newDist) return;
            if (__sync_bool_compare_and_swap(&ddata.dist, oldDist, newDist)) {
              ctx.push(WorkItem(dst, newDist + 1));
              break;
            }
          }
        }
      },
      galois::no_conflicts(),
      galois::loopname("bfsPushAsync"),
      galois::steal(),
      galois::wl<BSWL>()
    );
  }

  /**
   * Pull style BFS. Nodes update themselves if distance to that node is
   * equal to newDist
   *
   * @param graph Graph object
   * @param nextBag object that will contain all nodes that updated themselves
   * in this call
   * @param newDist a node updates itself to this if it has an incoming edge
   * from a source with newDist - 1 as its distance
   */
  void bfsPullTopo(Graph& graph, NodeBag* nextBag, const DistType& newDist) {
    galois::do_all(
      galois::iterate(graph),
      // Loop over in-edges: if the source node's dist + 1 is equivalent
      // to the the current new dist, "pull" and update self with new dist
      // (i.e. an edge exists to this node)
      [&, outer=this] (const GNode& n) {
        NodeData& sdata = graph.getData(n, galois::MethodFlag::UNPROTECTED);

        if (sdata.dist <= newDist)
          return;

        for (auto ii = graph.in_edge_begin(n, galois::MethodFlag::UNPROTECTED),
                  ei = graph.in_edge_end(n, galois::MethodFlag::UNPROTECTED); 
             ii != ei;
             ++ii) {
          GNode dst = graph.getInEdgeDst(ii);
          NodeData& ddata = graph.getData(dst, galois::MethodFlag::UNPROTECTED);

          if (ddata.dist + 1 == newDist) {
            sdata.dist = newDist;
            nextBag->push(n);
            outer->count += 1 + 
              std::distance(graph.edge_begin(n, galois::MethodFlag::UNPROTECTED),
                            graph.edge_end(n, galois::MethodFlag::UNPROTECTED));
            break;
          }
        }
      },
      galois::steal(),
      galois::loopname("bfsPullTopo")
    );
  }


  void operator()(Graph& graph, const GNode& source) {
    int next = 0; // next node bag to process; flips between 0 and 1
    DistType newDist = 1;
    int numForward = 0; // number of push phases done
    int numBackward = 0; // number of pull phases done

    graph.getData(source).dist = 0;

    // First round of BFS, i.e. only the original source

    // Chooses between a push style call or a pull style call depending
    // on number of edges on the source
    // If number of out-edges is great, do a pull.
    // Else do a push
    if (std::distance(graph.edge_begin(source), graph.edge_end(source)) + 1 > 
          (long) graph.sizeEdges() / 20) {
      bfsPullTopo(graph, &bags[next], newDist);
      numBackward += 1;
    } else {
      galois::do_all(
        galois::iterate(
          graph.out_edges(source, galois::MethodFlag::UNPROTECTED).begin(), 
          graph.out_edges(source, galois::MethodFlag::UNPROTECTED).end()
        ),
        [&, outer=this] (auto ii) {
          outer->bfsPushBulkSyncOperator(ii, graph, &outer->bags[next], 
                                         newDist);
        },
        galois::steal(),
        galois::loopname("bfsPushBulkSync")
      );

      numForward += 1;
    }

    // Handle the rest of the BFS propagation by processing updated nodes
    // pushed to bags
    while (!bags[next].empty()) {
      // flip "old" and "new" node bags
      int cur = next;
      next = (cur + 1) & 1; // if 0, become 1, if 1, become 0

      size_t nextSize = count.reduce();
      count.reset();
      newDist++;

      if (nextSize > graph.sizeEdges() / 20) {
        // Dense number of updates = do a pull
        bfsPullTopo(graph, &bags[next], newDist);
        numBackward += 1;
      } else if (numForward < 10 && numBackward == 0) {
        // if haven't done many push phases and backward phase count is 0,
        // do push
        galois::do_all(
          galois::iterate(bags[cur]),
          [&, outer=this] (const GNode& n) {
            for (auto ii = graph.edge_begin(n, galois::MethodFlag::UNPROTECTED),
                      ei = graph.edge_end(n, galois::MethodFlag::UNPROTECTED); 
                 ii != ei;
                 ++ii) {
              outer->bfsPushBulkSyncOperator(ii, graph, &outer->bags[next],   
                                             newDist);
            }
          },
          galois::steal(),
          galois::loopname("bfsPushBulkSync")
        );
        numForward += 1;
      } else {
        // ASYNC BFS

        // create a work item bag based on what was updated last round
        WorkItemBag asyncBag;
        galois::do_all(
          galois::iterate(bags[cur]), 
          [&] (const GNode& n) {
            asyncBag.push(WorkItem(n, newDist));
          }
        );

        // do a push, asynchronous version (i.e. finish off the rest of 
        // BFS)
        bfsPushAsync(graph, asyncBag);
        break;
      }

      // bags[cur] becomes bags[next] next round
      bags[cur].clear();
    }
  }
};

#endif
