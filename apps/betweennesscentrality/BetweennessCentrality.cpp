#include "Galois/Launcher.h"
#include "Galois/Graphs/LCGraph.h"
#include "Galois/Galois.h"
#include "Galois/IO/gr.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <stack>
#include <queue>
#include <vector>

typedef Galois::Graph::FirstGraph<unsigned long long, int, false> GraphT;
typedef GraphT::GraphNode GNodeT;

//LCGraph should handle void edgedata
typedef Galois::Graph::LCGraph<unsigned long long, int, false> Graph;
typedef Graph::GraphNode GNode;

Graph* G;
int NumNodes;

GaloisRuntime::CPUSpaced<std::vector<double> >* CB;

void process(GNode& _req, Galois::Context<GNode>& lwl) {
  typedef Galois::Context<GNode>::ItAllocTy::rebind<GNode>::other GNodeAlloc;
  std::vector<GNode> SQ(NumNodes);

  std::vector<int> sigma(NumNodes);
  std::vector<int> d(NumNodes);

  typedef std::multimap<int,int> MMapTy;
  MMapTy P;
  
  int QPush = 0;
  int QAt = 0;

  std::cerr << ".";

  int req = G->getId(_req);

  sigma[req] = 1;
  d[req] = 1;

  SQ[QPush++] = _req;

  while (QAt != QPush) {
    GNode _v = SQ[QAt++];
    int v = G->getId(_v);
    for (Graph::neighbor_iterator ii = G->neighbor_begin(_v, Galois::Graph::NONE), ee = G->neighbor_end(_v, Galois::Graph::NONE); ii != ee; ++ii) {
      GNode _w = *ii;
      int w = G->getId(_w);
      if (!d[w]) {
	SQ[QPush++] = _w;
	d[w] = d[v] + 1;
      }
      if (d[w] == d[v] + 1) {
	sigma[w] = sigma[w] + sigma[v];
	P.insert(std::pair<int, int>(w,v));
      }
    }
  }

  std::vector<double> delta(NumNodes);
  while (QAt) {
    int w = G->getId(SQ[--QAt]);
    std::pair<MMapTy::iterator, MMapTy::iterator> ppp = P.equal_range(w);
    for (MMapTy::iterator ii = ppp.first, ee = ppp.second;
	 ii != ee; ++ii) {
      int v = ii->second;
      delta[v] = delta[v] + ((double)sigma[v]/(double)sigma[w])*(1.0 + delta[w]);
    }
    if (w != req) {
      if (CB->get().size() < (unsigned int)w + 1)
	CB->get().resize(w+1);
      CB->get()[w] += delta[w];
    }
  }
}


void merge(std::vector<double>& lhs, std::vector<double>& rhs) {
  if (lhs.size() < rhs.size())
    lhs.resize(rhs.size());
  for (unsigned int i = 0; i < rhs.size(); i++)
    lhs[i] += rhs[i];
}


int main(int argc, char** argv) {
  if (argc < 2) {
    std::cout << "Arguments: [-t threads] <input file>\n";
    return 1;
  }

  int inputFileAt = 1;
  int threads = 1;
  if (std::string("-t").compare(argv[1]) == 0) {
    inputFileAt = 3;
    threads = atoi(argv[2]);
  }

  std::cout << "\nLonestar Benchmark Suite v3.0\n"
	    << "Copyright (C) 2007, 2008, 2009, 2010 The University of Texas at Austin\n"
	    << "http://iss.ices.utexas.edu/lonestar/\n"
	    << "\n"
	    << "application: Betweenness Centrality (c++ version)\n"
	    << "\n";

  GraphT gt;
  Graph  g;
  G = &g;
  GaloisRuntime::CPUSpaced<std::vector<double> > cb(merge);
  CB = &cb;

  //readTxtFile(argv[inputFileAt]);
  Galois::IO::readFile_gr<GraphT, false>(argv[inputFileAt], &gt);
  G->createGraph(&gt);

  NumNodes = G->size();

  std::vector<GNode> wl;
  for (Graph::active_iterator ii = g.active_begin(), ee = g.active_end();
       ii != ee; ++ii)
    wl.push_back(*ii);
  
  Galois::setMaxThreads(threads);
  Galois::Launcher::startTiming();
#ifdef WITH_VTUNE
  __itt_resume();
#endif
  Galois::for_each(wl.begin(), wl.end(), process);
#ifdef WITH_VTUNE
  __itt_pause();
#endif
  Galois::Launcher::stopTiming();

  std::cout << "STAT: Time " << Galois::Launcher::elapsedTime() << "\n";


  return 0;
}
