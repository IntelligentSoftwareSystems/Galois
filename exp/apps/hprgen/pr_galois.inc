#include "hpr.h"

/*********************************************************************************
 *
 **********************************************************************************/
struct InitializeGraph {
   Graph* g;
   void static go(Graph& _g, unsigned num) {
      Galois::do_all(_g.begin(), _g.begin() + num, InitializeGraph { &_g }, Galois::loopname("init"));
   }
   void operator()(GNode src) const {
      LNode& sdata = g->getData(src);
      sdata.pr = 1.0 - alpha;
      buffered_updates[src] = 0;
      for (auto nbr = g->edge_begin(src); nbr != g->edge_end(src); ++nbr) {
         __sync_fetch_and_add(&g->getData(g->getEdgeDst(*nbr)).nout, 1);
      }
   }
};

/*********************************************************************************
 * CPU PageRank operator implementation.
 **********************************************************************************/
struct WriteBack {
   Graph * g;
   void static go(Graph& _g, unsigned num) {
      Galois::do_all(_g.begin(), _g.begin() + num, WriteBack { &_g }, Galois::loopname("Writeback"));
   }
   void operator()(GNode src) const {
      LNode& sdata = g->getData(src);
      sdata.pr = buffered_updates[src];
      buffered_updates[src] = 0;
   }
};
struct PageRank {
   Graph* g;

   void static go(Graph& _g, unsigned num) {
      Galois::do_all(_g.begin(), _g.begin() + num, PageRank { &_g }, Galois::loopname("Page Rank"));
   }

   void operator()(GNode src) const {
      double sum = 0;
      LNode& sdata = g->getData(src);
      for (auto jj = g->edge_begin(src), ej = g->edge_end(src); jj != ej; ++jj) {
         GNode dst = g->getEdgeDst(jj);
         LNode& ddata = g->getData(dst);
         if(ddata.nout!=0)
         sum += ddata.pr / ddata.nout;
      }
      float value = (1.0 - alpha) * sum + alpha;
      float diff = std::fabs(value - sdata.pr);
//      sdata.pr = value;
      buffered_updates[src] = value;
   }
};
