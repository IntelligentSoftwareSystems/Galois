/** Page rank application -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
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
 * @author Gurbinder Gill <gill@cs.utexas.edu>
 */


#include "Galois/Galois.h"
#include "Galois/gstl.h"
#include "Galois/Graphs/LC_Dist_Graph.h"
#include "Galois/Graph/FileGraph.h"
#include "Galois/Graphs/LC_Dist_InOut_Graph.h"
#include "Galois/Bag.h"


#include "Lonestar/BoilerPlate.h"

#include <iostream>
#include <typeinfo>

static const char* const name = "Page Rank - Distributed";
static const char* const desc = "Computes PageRank on Distributed Galois";
static const char* const url = 0;


namespace cll = llvm::cl;
static cll::opt<std::string> inputFile (cll::Positional, cll::desc("<input file>"), cll::Required);
static cll::opt<unsigned int> maxIterations ("maxIterations", cll::desc("Maximum iterations"), cll::init(1));

static int TOTAL_NODES;

struct LNode {
    float value;
    unsigned int nout;
    LNode() : value(1.0), nout(0) {}
    float getPageRank() { return value; }
    
   typedef int  tt_is_copyable;
};

typedef Galois::Graph::LC_Dist_InOut<LNode, int> Graph;
typedef typename Graph::GraphNode GNode;

// Constants for page Rank Algo.
//! d is the damping factor. Alpha is the prob that user will do a random jump, i.e., 1 - d
static const double alpha = (1.0 - 0.85);

//! maximum relative change until we deem convergence
static const double TOLERANCE = 0.1;


struct InitializeGraph {
    Graph::pointer g;
//    InitializeGraph(Graph::pointer _g) : g(_g) { }

    void static go(Graph::pointer g)
    {
      Galois::for_each_local(g, InitializeGraph{g}, Galois::loopname("init"));
    }

    void operator()(GNode n, Galois::UserContext<GNode>& cnx) const {
      LNode& data = g->at(n);
      data.value = 1.0;
      //Adding Galois::NONE is imp. here since we don't need any blocking locks.
      data.nout = std::distance(g->in_edge_begin(n, Galois::MethodFlag::SRC_ONLY),g->in_edge_end(n, Galois::MethodFlag::SRC_ONLY));	
    }

  typedef int tt_is_copyable;
};


struct PageRank {
    Graph::pointer g;
    void static go(Graph::pointer g)
    {
      Galois::Timer round_time;
      for(int iterations = 0; iterations < maxIterations; ++iterations){
          round_time.start();
          Galois::for_each_local(g, PageRank{g}, Galois::loopname("Page Rank"));
          round_time.stop();
          std::cout<<"Iteration : " << iterations << "  Time : " << round_time.get() << "ms\n";
      }
    }

    void operator() (GNode src, Galois::UserContext<GNode>& cnx) const {
      double sum = 0;
      LNode& n = g->at(src);
      //std::cout << "n :" << n.nout <<"\n";
      for (auto jj = g->in_edge_begin(src, Galois::MethodFlag::ALL), ej = g->in_edge_end(src, Galois::MethodFlag::SRC_ONLY); jj != ej; ++jj) {
        GNode dst = g->dst(jj, Galois::MethodFlag::SRC_ONLY);
        LNode& ddata = g->at(dst, Galois::MethodFlag::SRC_ONLY);
        sum += ddata.value / ddata.nout;
      }
      float value = (1.0 - alpha) * sum + alpha;
      LNode&  sdata = g->at(src, Galois::MethodFlag::SRC_ONLY);
      float diff = std::fabs(value - sdata.value);
      if (diff > TOLERANCE) {
        sdata.value = value;
        /*   for (auto jj = g->edge_begin(src, Galois::MethodFlag::SRC_ONLY), ej = g->edge_end(src, Galois::MethodFlag::SRC_ONLY); jj != ej; ++jj) {
             GNode dst = g->dst(jj, Galois::MethodFlag::SRC_ONLY);
             cnx.push(dst);
             }
             */
      }

  }

  typedef int tt_is_copyable;
};

/* 
 * collect page rank of all the nodes 
 * */

/*struct compute_total_rank {
    Graph::pointer g;
    int total_rank = 0;
    std::vector<int> a[TOTAL_NODES];
    
    void static go(Graph::pointer g) {
	Galois::for_each_local(g, compute_total_rank(g), Galois::loopname("compute total rank")i);
    }

    void operator() (GNode n, Galois::UserContext<GNode>& cnx) const {
	LNode& n = 
    }

    for(auto ii = g->begin(), ei = g->end(); ii != ei ; ++ii) {
	//LNode& ddata = g->at(ii, Galois::MethodFlag::SRC_ONLY);
	//total_rank += ddata.value;
    }

    
    //return total_rank;
};
*/

int compute_total_rank(Graph::pointer g) {
    int total_rank = 0;

    for(auto ii = g->begin(), ei=g->end(); ii != ei; ++ii) {
      LNode& node = g->at(*ii); 
      total_rank += node.value;
    }


    return total_rank;

}

int main(int argc, char** argv) {
    LonestarStart (argc, argv, name, desc, url);
    Galois::StatManager statManager;

    Galois::Timer timerLoad;
    timerLoad.start();
    
    //allocate local computation graph and Reading from the inputFile using FileGraph
    //NOTE: We are computing in edges on the fly and then using then in Graph construction.
    Graph::pointer g;
    {
      Galois::Graph::FileGraph fg;
      fg.fromFile(inputFile);
      std::vector<unsigned> counts;
      std::vector<unsigned> In_counts;
      for(auto& N : fg)
      {
          //std::cout << "N = " << N << "   : \n";
          counts.push_back(std::distance(fg.edge_begin(N), fg.edge_end(N)));
          for(auto ii = fg.edge_begin(N), ei = fg.edge_end(N); ii != ei; ++ii)
          {
              unsigned dst = fg.getEdgeDst(ii);
              //std::cout << dst << " , " ;
              if(dst >= In_counts.size()) {
                // +1 is imp because vec.resize makes sure new vec can hold dst entries so it 
                // will not have vec[dst] which is (dst+1)th entry!!
                In_counts.resize(dst+1); 
              }
              In_counts[dst]+=1;
          }
          //std::cout << "\n";
      }
      if(counts.size() >  In_counts.size())
          In_counts.resize(counts.size());

      TOTAL_NODES = counts.size();

      std::cout << "size of transpose : " << In_counts.size() <<" : : "<<In_counts[0] <<"\n";
      std::cout << "size of counts : " << counts.size() << "\n";
      g = Graph::allocate(counts, In_counts);

      for (unsigned x = 0; x < counts.size(); ++x) {
        auto fgn = *(fg.begin() + x);
        auto gn = *(g->begin() + x);
        //std::cout << "x = " << x << "   : \n";
        for (auto ii = fg.edge_begin(fgn), ee = fg.edge_end(fgn); ii != ee; ++ii) {
          unsigned dst = fg.getEdgeDst(ii);
          //std::cout <<"Incount["<<dst<<"]"<<In_counts[dst]<< " , " ;
          int val = fg.getEdgeData<int>(ii);
          g->addEdge(gn, *(g->begin() + dst), val, Galois::MethodFlag::SRC_ONLY);
          g->addInEdge(*(g->begin() + dst),gn, val, Galois::MethodFlag::SRC_ONLY);
        }
              //std::cout << "\n";
    }

  }
    //Graph Construction ENDS here.
    timerLoad.stop();
    std::cout << "Graph Loading: " << timerLoad.get() << " ms\n";

    //Graph Initialization begins.
    Galois::Timer timerInit;
    timerInit.start();
    //Galois::for_each_local(g,InitializeGraph{}, Galois::loopname("init"));

    InitializeGraph::go(g);

    //auto N_n = *(g->begin() + 67);
    //LNode& n = g->at(N_n);
    //std::cout << "n.out : " << n.nout << "\n";
    timerInit.stop();
    std::cout << "Graph Initialization: " << timerInit.get() << " ms\n";

    Galois::Timer timerPR;
    timerPR.start();
    //typedef Galois::WorkList::dChunkedFIFO<512> WL;
    //Galois::for_each(g->begin(), g->end(), PageRank{g}, Galois::wl<WL>());

    PageRank::go(g);

    timerPR.stop();
    std::cout << "Page Rank: " << timerPR.get() << " ms\n";
    std::cout << "Total Page Rank: " << compute_total_rank(g) << "\n";

 /* 
   std::cout << "size = " << g->size() <<std::endl;
   std::cout << "Typeinfo " << typeid(Graph::GraphNode).name() <<std::endl;

/////Checking Graph /////
   //for (auto ii = g->begin(), ee = g->end(); ii != ee; ++ii) {
   	auto N = *(g->begin() + 0);
	auto N_1 = *(g->begin() + 1);
	auto N_9 = *(g->begin() + 9);
	
	LNode& data_1 = g->at(N_1);
	data_1.value = 10;

	LNode& data_9 = g->at(N_9);
	data_9.value = 19;
*/	
/*	for (auto jj = g->in_edge_begin(N), ej = g->in_edge_end(N); jj != ej; ++jj) {
	    std::cout << "Inside : ";
	    GNode dst = g->dst(jj);
	    LNode& data = g->at(dst);
	    data.value = 9;
	    //std::cout << data.value << "\n";
	}
*/
/*	for (auto jj = g->in_edge_begin(N), ej = g->in_edge_end(N); jj != ej; ++jj) {
	    std::cout << "Inside2 : ";
	    GNode dst = g->dst(jj);
	    LNode& data = g->at(dst);
	    std::cout << data.value << "\n";
	}
*/ 
//    }
       // typedef Galois::WorkList::dChunkedFIFO<512> WL;
       Galois::Runtime::getSystemNetworkInterface().terminate();
    return 0;
}





