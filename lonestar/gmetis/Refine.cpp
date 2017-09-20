/** GMetis -*- C++ -*-
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
 * @author Xin Sui <xinsui@cs.utexas.edu>
 * @author Nikunj Yadav <nikunj@cs.utexas.edu>
 * @author Andrew Lenharth <andrew@lenharth.org>
 */
#include "Galois/Galois.h"
#include "Galois/Accumulator.h"
#include "Galois/Timer.h"
#include "Metis.h"
#include <set>
#include <iostream>

namespace {

struct gainIndexer : public std::unary_function<GNode, int> {
  static GGraph* g;

  int operator()(GNode n) {
    int retval = 0;
    galois::MethodFlag flag = galois::MethodFlag::UNPROTECTED;
    unsigned int nPart = g->getData(n, flag).getPart();
    for (auto ii = g->edge_begin(n, flag), ee = g->edge_end(n); ii != ee; ++ii) {
      GNode neigh = g->getEdgeDst(ii);
      if (g->getData(neigh, flag).getPart() == nPart)
        retval -= g->getEdgeData(ii, flag);
      else
        retval += g->getEdgeData(ii, flag);
    }
    return -retval / 16;
  }
};

GGraph* gainIndexer::g;

bool isBoundary(GGraph& g, GNode n) {
  unsigned int nPart = g.getData(n).getPart();
  for (auto ii : g.edges(n))
    if (g.getData(g.getEdgeDst(ii)).getPart() != nPart)
      return true;
  return false;
}

//This is only used on the terminal graph (find graph)
struct findBoundary {
  galois::InsertBag<GNode>& b;
  GGraph& g;
  findBoundary(galois::InsertBag<GNode>& _b, GGraph& _g) : b(_b), g(_g) {}
  void operator()(GNode n) const {
    auto& cn = g.getData(n, galois::MethodFlag::UNPROTECTED);
    if (cn.getmaybeBoundary())
      cn.setmaybeBoundary(isBoundary(g,n));
    if (cn.getmaybeBoundary())
      b.push(n);
  }
};

//this is used on the coarse graph to project to the fine graph
struct findBoundaryAndProject {
  galois::InsertBag<GNode>& b;
  GGraph& cg;
  GGraph& fg;
  findBoundaryAndProject(galois::InsertBag<GNode>& _b, GGraph& _cg, GGraph& _fg) :b(_b), cg(_cg), fg(_fg) {}
  void operator()(GNode n) const {
    auto& cn = cg.getData(n, galois::MethodFlag::UNPROTECTED);
    if (cn.getmaybeBoundary())
      cn.setmaybeBoundary(isBoundary(cg,n));

    //project part and maybe boundary
    //unsigned part = cn.getPart();
    for (unsigned x = 0; x < cn.numChildren(); ++x) {
      fg.getData(cn.getChild(x), galois::MethodFlag::UNPROTECTED).initRefine(cn.getPart(), cn.getmaybeBoundary());
    }
    if (cn.getmaybeBoundary())
      b.push(n);
  }
};


template<bool balance>
struct refine_BKL2 {
  unsigned meanSize;
  unsigned minSize;
  unsigned maxSize;
  GGraph& cg;
  GGraph* fg;
  std::vector<partInfo>& parts;

  typedef int tt_needs_per_iter_alloc;

  refine_BKL2(unsigned mis, unsigned mas, GGraph& _cg, GGraph* _fg, std::vector<partInfo>& _p) : minSize(mis), maxSize(mas), cg(_cg), fg(_fg), parts(_p) {}

  //Find the partition n is most connected to
  template<typename Context>
  unsigned pickPartitionEC(GNode n, Context& cnx) {
    std::vector<unsigned, galois::PerIterAllocTy::rebind<unsigned>::other> edges(parts.size(), 0, cnx.getPerIterAlloc());
    unsigned P = cg.getData(n).getPart();
    for (auto ii : cg.edges(n)) {
      GNode neigh = cg.getEdgeDst(ii);
      auto& nd = cg.getData(neigh);
      if (parts[nd.getPart()].partWeight < maxSize
          || nd.getPart() == P)
        edges[nd.getPart()] += cg.getEdgeData(ii);
    }
    return std::distance(edges.begin(), std::max_element(edges.begin(), edges.end()));
  }

  //Find the smallest partition n is connected to
  template<typename Context>
  unsigned pickPartitionMP(GNode n, Context& cnx) {
    unsigned P = cg.getData(n).getPart();
    unsigned W = parts[P].partWeight;
    std::vector<unsigned, galois::PerIterAllocTy::rebind<unsigned>::other> edges(parts.size(), ~0, cnx.getPerIterAlloc());
     edges[P] = W;
    W = (double)W * 0.9;
    for (auto ii : cg.edges(n)) {
      GNode neigh = cg.getEdgeDst(ii);
      auto& nd = cg.getData(neigh);
      if (parts[nd.getPart()].partWeight < W)
        edges[nd.getPart()] = parts[nd.getPart()].partWeight;
    }
    return std::distance(edges.begin(), std::min_element(edges.begin(), edges.end()));
  }


  template<typename Context>
  void operator()(GNode n, Context& cnx) {
    auto& nd = cg.getData(n);
    unsigned curpart = nd.getPart();
    unsigned newpart = balance ? pickPartitionMP(n, cnx) : pickPartitionEC(n, cnx);
    if(parts[curpart].partWeight < minSize) return;
    if (curpart != newpart) {
      nd.setPart(newpart);
      __sync_fetch_and_sub(&parts[curpart].partWeight, nd.getWeight());
      __sync_fetch_and_add(&parts[newpart].partWeight, nd.getWeight());
      for (auto ii : cg.edges(n)) {
        GNode neigh = cg.getEdgeDst(ii);
        auto& ned = cg.getData(neigh);
        if (ned.getPart() != newpart && !ned.getmaybeBoundary()) {
          ned.setmaybeBoundary(true);
          if (fg)
            for (unsigned x = 0; x < ned.numChildren(); ++x)
              fg->getData(ned.getChild(x), galois::MethodFlag::UNPROTECTED).setmaybeBoundary(true);
        }
        //if (ned.getPart() != newpart)
        //cnx.push(neigh);
      }
      if (fg)
        for (unsigned x = 0; x < nd.numChildren(); ++x)
          fg->getData(nd.getChild(x), galois::MethodFlag::UNPROTECTED).setPart(newpart);
    }
  }

  static void go(unsigned mins, unsigned maxs, GGraph& cg, GGraph* fg,  std::vector<partInfo>& p) {
    typedef galois::WorkList::dChunkedFIFO<8> Chunk;
    typedef galois::WorkList::OrderedByIntegerMetric<gainIndexer, Chunk, 10> pG;
    gainIndexer::g = &cg;
    galois::InsertBag<GNode> boundary;
    if (fg)
      galois::do_all_local(cg, findBoundaryAndProject(boundary, cg, *fg), galois::loopname("boundary"));
    else
      galois::do_all_local(cg, findBoundary(boundary, cg), galois::loopname("boundary"));
    galois::for_each_local(boundary, refine_BKL2(mins, maxs, cg, fg, p), galois::loopname("refine"), galois::wl<pG>());
    if (false) {
      galois::InsertBag<GNode> boundary;
      galois::do_all_local(cg, findBoundary(boundary, cg), galois::loopname("boundary"));
      galois::for_each_local(boundary, refine_BKL2(mins, maxs, cg, fg, p), galois::loopname("refine"), galois::wl<pG>());
    }

  }
};

struct projectPart {
  GGraph* fineGraph;
  GGraph* coarseGraph;
  std::vector<partInfo>& parts;

  projectPart(MetisGraph* Graph, std::vector<partInfo>& p) :fineGraph(Graph->getFinerGraph()->getGraph()), coarseGraph(Graph->getGraph()), parts(p) {}

  void operator()(GNode n) const {
    auto& cn = coarseGraph->getData(n);
    unsigned part = cn.getPart();
    for (unsigned x = 0; x < cn.numChildren(); ++x)
      fineGraph->getData(cn.getChild(x)).setPart(part);
  }

  static void go(MetisGraph* Graph, std::vector<partInfo>& p) {
    galois::do_all_local(*Graph->getGraph(), projectPart(Graph, p), galois::loopname("project"));
  }
};

} //anon namespace




int gain(GGraph& g, GNode n) {
  int retval = 0;
  unsigned int nPart = g.getData(n).getPart();
  for (auto ii : g.edges(n)) {
    GNode neigh = g.getEdgeDst(ii);
    if (g.getData(neigh).getPart() == nPart)
      retval -= g.getEdgeData(ii);
    else
      retval += g.getEdgeData(ii);
  }
  return retval;
}

struct parallelBoundary {
  galois::InsertBag<GNode> &bag;
  GGraph& g;
  parallelBoundary(galois::InsertBag<GNode> &bag, GGraph& graph):bag(bag),g(graph) {

  }
  void operator()(GNode n,galois::UserContext<GNode>&ctx) {
      if (gain(g,n) > 0)
        bag.push(n);
  }
};
void refineOneByOne(GGraph& g, std::vector<partInfo>& parts) {
  std::vector<GNode>  boundary;
  unsigned int meanWeight =0;
  for (unsigned int i =0; i<parts.size(); i++)
    meanWeight += parts[i].partWeight;
  meanWeight /= parts.size();
  galois::InsertBag<GNode> boundaryBag;
  parallelBoundary pB(boundaryBag, g);
  galois::for_each(g.begin(), g.end(), pB, galois::loopname("Get Boundary"));

  for (auto ii = boundaryBag.begin(), ie =boundaryBag.end(); ii!=ie;ii++){
      GNode n = (*ii) ;
      unsigned nPart = g.getData(n).getPart();
      int part[parts.size()];
      for (unsigned int i =0; i<parts.size(); i++)part[i]=0;
      for (auto ii : g.edges(n)) {
        GNode neigh = g.getEdgeDst(ii);
        part[g.getData(neigh).getPart()]+=g.getEdgeData(ii);
      }
      int t = part[nPart];
      unsigned int p = nPart;
      for (unsigned int i =0; i<parts.size(); i++)
        if (i!=nPart && part[i] > t && parts[nPart].partWeight>  parts[i].partWeight*(98)/(100) && parts[nPart].partWeight > meanWeight*98/100){
          t = part[i];
          p = i;
        }
    if(p != nPart){
      g.getData(n).setPart(p);
      parts[p].partWeight += g.getData(n).getWeight();
      parts[nPart].partWeight -= g.getData(n).getWeight();
    }
  }
}


void refine_BKL(GGraph& g, std::vector<partInfo>& parts) {
  std::set<GNode> boundary;

  //find boundary nodes with positive gain
  galois::InsertBag<GNode> boundaryBag;
  parallelBoundary pB(boundaryBag, g);
  galois::for_each(g.begin(), g.end(), pB, galois::loopname("Get Boundary"));
  for (auto ii = boundaryBag.begin(), ie =boundaryBag.end(); ii!=ie;ii++ ){
    boundary.insert(*ii);}

  //refine by swapping with a neighbor high-gain node
  while (!boundary.empty()) {
    GNode n = *boundary.begin();
    boundary.erase(boundary.begin());
    unsigned nPart = g.getData(n).getPart();
    for (auto ii : g.edges(n)) {
      GNode neigh = g.getEdgeDst(ii);
      unsigned neighPart = g.getData(neigh).getPart();
      if (neighPart != nPart && boundary.count(neigh) &&
          gain(g, n) > 0 && gain(g, neigh) > 0 ) {
        unsigned nWeight = g.getData(n).getWeight();
        unsigned neighWeight = g.getData(neigh).getWeight();
        //swap
        g.getData(n).setPart(neighPart);
        g.getData(neigh).setPart(nPart);
        //update partinfo
        parts[neighPart].partWeight += nWeight;
        parts[neighPart].partWeight -= neighWeight;
        parts[nPart].partWeight += neighWeight;
        parts[nPart].partWeight -= nWeight;
        //remove nodes
        boundary.erase(neigh);
        break;
      }
    }
  }
}

struct ChangePart {//move each node to its nearest cluster
  GGraph& g;
  int nbCluster;
  double* Dist;
  int* card;

  ChangePart(GGraph& g, int nb_cluster, double* Dist, int* card): g(g), nbCluster(nb_cluster), Dist(Dist), card(card){
  }

  void operator()(GNode n, galois::UserContext<GNode>& ctx) {
    double dmin = std::numeric_limits<double>::min();
    int partition =-1;
    std::map <int, int> degreein;
    degreein[g.getData(n, galois::MethodFlag::UNPROTECTED).getOldPart()] +=1;
    for (auto ii : g.edges(n, galois::MethodFlag::UNPROTECTED)) {
      int nclust = g.getData(g.getEdgeDst(ii), galois::MethodFlag::UNPROTECTED).getOldPart();
      degreein[nclust] += (int) g.getEdgeData(ii, galois::MethodFlag::UNPROTECTED);
    }

    for(auto clust = degreein.begin(), ee = degreein.end(); clust != ee; clust++)
    {
      //the distance between the cluster clust and the noden is :
      double d = Dist[clust->first]-(2.0*(double)clust->second/(double)card[clust->first]);
      if(d < dmin || partition ==-1)
      {
        dmin = d;
        partition = clust->first;
      }
    }
    g.getData(n, galois::MethodFlag::UNPROTECTED).setPart(partition);
  }


};

 // galois::GAccumulator<size_t> count
struct ComputeClusterDist {
  GGraph& g;
  int nbCluster;
  galois::GAccumulator<size_t> *card;
  galois::GAccumulator<size_t> *degreeIn;

  ComputeClusterDist(GGraph& g, int nb_cluster): g(g), nbCluster(nb_cluster) {
    card = new galois::GAccumulator<size_t>[nbCluster];
    degreeIn = new galois::GAccumulator<size_t>[nbCluster];
  }

  /*~ComputeClusterDist(){
    std::cout <<"destruct\n"; delete[] card; delete[] degreeIn;
  }*/

  void operator()(GNode n, galois::UserContext<GNode>& ctx) {
    unsigned int clust = g.getData(n, galois::MethodFlag::UNPROTECTED).getPart();
    int degreet =0;

    g.getData(n, galois::MethodFlag::UNPROTECTED).OldPartCpyNew();
    for (auto ii : g.edges(n, galois::MethodFlag::UNPROTECTED)) 
      if (g.getData(g.getEdgeDst(ii), galois::MethodFlag::UNPROTECTED).getPart() == clust)
        degreet+=(int) g.getEdgeData(ii, galois::MethodFlag::UNPROTECTED);
    card[clust]+=g.getData(n, galois::MethodFlag::UNPROTECTED).getWeight();
    degreeIn[clust] += degreet;
  }
};
double ratiocut(int nbClust, int* degree, int* card)
{
  double res=0;
  for (int i=0; i<nbClust;i++)
    res += (double)(degree[i])/(double)(card[i]);

  return res;
}



void GraclusRefining(GGraph* graph, int nbParti, int nbIter)
{

  nbIter = std::min(15, nbIter);
  double Dist[nbParti];
  int card[nbParti];
  int degreeIn[nbParti];


  for(int j=0;j<nbIter;j++)
  {
    galois::StatTimer T3("1st loop");
    T3.start();
    ComputeClusterDist comp(*graph, nbParti);
    galois::for_each(graph->begin(), graph->end(), comp, galois::loopname("compute dists"));
    T3.stop();
    //std::cout << "Time calc:  "<<T3.get()<<'\n';

    for (int i=0; i<nbParti; i++)
    {
      card[i] = comp.card[i].reduce();
      Dist[i] = (card[i]!=0)?(double)((degreeIn[i]= comp.degreeIn[i].reduce())+card[i] )/((double)card[i]*(double)card[i]) : 0;
    }
    delete[] comp.card; delete[] comp.degreeIn;
    galois::StatTimer T4("2nd loop");
    T4.start();

    galois::for_each(graph->begin(), graph->end(), ChangePart(*graph, nbParti, Dist, card), galois::loopname("make moves"));
    T4.stop();
    //std::cout << "Time move:  "<<T4.get()<<'\n';
  }
  /*  std::cout<<ratiocut(nbParti, degreeIn, card)<< '\n';
  for (int i=0; i<nbParti; i++)
    std::cout<<card[i]<< ' ';
  std::cout<<std::endl;*/

}



void refine(MetisGraph* coarseGraph, std::vector<partInfo>& parts, unsigned minSize, unsigned maxSize,
            refinementMode refM, bool verbose) {
  MetisGraph* tGraph = coarseGraph;
  int nbIter=1;
  if (refM == GRACLUS) {
    while ((tGraph = tGraph->getFinerGraph())) nbIter*=2;
    nbIter /=4;
  }
  do {
    MetisGraph* fineGraph = coarseGraph->getFinerGraph();
    bool doProject = true;
    if (verbose) { 
      std::cout << "Cut " << computeCut(*coarseGraph->getGraph()) << " Weights ";
      printPartStats(parts);
      std::cout << "\n";
    }
    //refine nparts times
    switch (refM) {
    case BKL2: refine_BKL2<false>::go(minSize, maxSize, *coarseGraph->getGraph(), fineGraph ? fineGraph->getGraph() : nullptr, parts); doProject = false; break;
    case BKL: refine_BKL(*coarseGraph->getGraph(), parts); break;
    case ROBO: refineOneByOne(*coarseGraph->getGraph(), parts); break;
    case GRACLUS: GraclusRefining(coarseGraph->getGraph(), parts.size(), nbIter);nbIter =(nbIter+1)/2;break;
    default: abort();
    }
    //project up
    if (fineGraph && doProject) {
      projectPart::go(coarseGraph, parts);
    }
  } while ((coarseGraph = coarseGraph->getFinerGraph()));
}

/*
void balance(MetisGraph* coarseGraph, std::vector<partInfo>& parts, unsigned meanSize) {
    MetisGraph* fineGraph = coarseGraph->getFinerGraph();
    refine_BKL2<true>::go(meanSize, *coarseGraph->getGraph(), fineGraph ? fineGraph->getGraph() : nullptr, parts);
}
*/

