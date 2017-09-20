#ifndef APPS_BFS_HYBRIDBFS_H
#define APPS_BFS_HYBRIDBFS_H

#include "Galois/Galois.h"
#include "Galois/Graphs/LCGraph.h"

template<typename NodeData,typename Dist>
struct HybridBFS {
  typedef typename galois::Graph::LC_CSR_Graph<NodeData,void>
    ::template with_no_lockable<true>::type
    ::template with_numa_alloc<true>::type 
    InnerGraph;
  typedef typename galois::Graph::LC_InOut_Graph<InnerGraph> Graph;
  typedef typename Graph::GraphNode GNode;

  typedef std::pair<GNode,Dist> WorkItem;
  typedef galois::InsertBag<GNode> NodeBag;
  typedef galois::InsertBag<WorkItem> WorkItemBag;
  
  galois::GAccumulator<size_t> count;
  NodeBag bags[2];

  struct ForwardProcess {
    typedef int tt_does_not_need_aborts;

    Graph& graph;
    HybridBFS* self;
    NodeBag* nextBag;
    Dist newDist;

    ForwardProcess(Graph& g, HybridBFS* s, NodeBag* n = 0, int d = 0):
      graph(g), self(s), nextBag(n), newDist(d) { }

    void operator()(const GNode& n, galois::UserContext<GNode>&) {
      for (typename Graph::edge_iterator ii = graph.edge_begin(n, galois::MethodFlag::UNPROTECTED),
            ei = graph.edge_end(n, galois::MethodFlag::UNPROTECTED); ii != ei; ++ii) {
        processBS(ii, newDist, *nextBag);
      }
    }

    void operator()(const typename Graph::edge_iterator& ii, galois::UserContext<typename Graph::edge_iterator>&) {
      processBS(ii, newDist, *nextBag);
    }

    void operator()(const WorkItem& item, galois::UserContext<WorkItem>& ctx) {
      GNode n = item.first;
      for (typename Graph::edge_iterator ii = graph.edge_begin(n, galois::MethodFlag::UNPROTECTED),
            ei = graph.edge_end(n, galois::MethodFlag::UNPROTECTED); ii != ei; ++ii) {
        processAsync(ii, item.second, ctx);
      }
    }

    void processBS(const typename Graph::edge_iterator& ii, Dist nextDist, NodeBag& next) {
      GNode dst = graph.getEdgeDst(ii);
      NodeData& ddata = graph.getData(dst, galois::MethodFlag::UNPROTECTED);

      Dist oldDist;
      while (true) {
        oldDist = ddata.dist;
        if (oldDist <= nextDist)
          return;
        if (__sync_bool_compare_and_swap(&ddata.dist, oldDist, nextDist)) {
          next.push(dst);
          self->count += 1 
            + std::distance(graph.edge_begin(dst, galois::MethodFlag::UNPROTECTED),
              graph.edge_end(dst, galois::MethodFlag::UNPROTECTED));
          break;
        }
      }
    }

    void processAsync(const typename Graph::edge_iterator& ii, Dist nextDist, galois::UserContext<WorkItem>& next) {
      GNode dst = graph.getEdgeDst(ii);
      NodeData& ddata = graph.getData(dst, galois::MethodFlag::UNPROTECTED);

      Dist oldDist;
      while (true) {
        oldDist = ddata.dist;
        if (oldDist <= nextDist)
          return;
        if (__sync_bool_compare_and_swap(&ddata.dist, oldDist, nextDist)) {
          next.push(WorkItem(dst, nextDist + 1));
          break;
        }
      }
    }
  };

  struct BackwardProcess {
    typedef int tt_does_not_need_aborts;
    typedef int tt_does_not_need_push;

    Graph& graph;
    HybridBFS* self;
    NodeBag* nextBag;
    Dist newDist; 
    BackwardProcess(Graph& g, HybridBFS* s, NodeBag* n, int d): graph(g), self(s), nextBag(n), newDist(d) { }

    void operator()(const GNode& n, galois::UserContext<GNode>&) {
      (*this)(n);
    }

    void operator()(const GNode& n) const {
      NodeData& sdata = graph.getData(n, galois::MethodFlag::UNPROTECTED);
      if (sdata.dist <= newDist)
        return;

      for (typename Graph::in_edge_iterator ii = graph.in_edge_begin(n, galois::MethodFlag::UNPROTECTED),
            ei = graph.in_edge_end(n, galois::MethodFlag::UNPROTECTED); ii != ei; ++ii) {
        GNode dst = graph.getInEdgeDst(ii);
        NodeData& ddata = graph.getData(dst, galois::MethodFlag::UNPROTECTED);

        if (ddata.dist + 1 == newDist) {
          sdata.dist = newDist;
          nextBag->push(n);
          self->count += 1
            + std::distance(graph.edge_begin(n, galois::MethodFlag::UNPROTECTED),
              graph.edge_end(n, galois::MethodFlag::UNPROTECTED));
          break;
        }
      }
    }
  };

  struct PopulateAsync {
    typedef int tt_does_not_need_aborts;
    typedef int tt_does_not_need_push;

    WorkItemBag& bag;
    Dist newDist;
    PopulateAsync(WorkItemBag& b, Dist d): bag(b), newDist(d) { }
    void operator()(const GNode& n, galois::UserContext<GNode>&) {
      (*this)(n);
    }

    void operator()(const GNode& n) {
      bag.push(WorkItem(n, newDist));
    }
  };

  void operator()(Graph& graph, const GNode& source) {
    using namespace galois::WorkList;
    typedef dChunkedLIFO<256> WL;
    typedef BulkSynchronous<dChunkedLIFO<256> > BSWL;

    int next = 0;
    Dist newDist = 1;
    int numForward = 0;
    int numBackward = 0;

    graph.getData(source).dist = 0;
    if (std::distance(graph.edge_begin(source), graph.edge_end(source)) + 1 > (long) graph.sizeEdges() / 20) {
      galois::do_all_local(graph, BackwardProcess(graph, this, &bags[next], newDist));
      numBackward += 1;
    } else {
      galois::for_each(graph.out_edges(source, galois::MethodFlag::UNPROTECTED).begin(), 
          graph.out_edges(source, galois::MethodFlag::UNPROTECTED).end(),
          ForwardProcess(graph, this, &bags[next], newDist));
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
        galois::do_all_local(graph, BackwardProcess(graph, this, &bags[next], newDist));
        numBackward += 1;
      } else if (numForward < 10 && numBackward == 0) {
        //std::cout << "Sparse " << nextSize << "\n";
        galois::for_each_local(bags[cur], ForwardProcess(graph, this, &bags[next], newDist), galois::wl<WL>());
        numForward += 1;
      } else {
        //std::cout << "Async " << nextSize << "\n";
        WorkItemBag asyncBag;
        galois::for_each_local(bags[cur], PopulateAsync(asyncBag, newDist), galois::wl<WL>());
        galois::for_each_local(asyncBag, ForwardProcess(graph, this), galois::wl<BSWL>());
        break;
      }
      bags[cur].clear();
    }
  }
};

#endif
