
#include "Metis.h"
#include "galois/Galois.h"
#include "galois/AtomicHelpers.h"
#include "galois/Reduction.h"
#include "galois/runtime/Profile.h"
#include "galois/substrate/PerThreadStorage.h"
#include "galois/gstl.h"

#include <iostream>
#include <fstream>
#include <string>
#include <unordered_map>

void normalize(MetisGraph* mcg, std::string filename) {
  GGraph* g = mcg->getGraph();
  int nsize = std::distance(g->cellList().begin(), g->cellList().end());
  int hsize = std::distance(g->getNets().begin(), g->getNets().end());
  std::cout<<"in norm\n";
  std::map<int, GNode> maps;
  for (auto c : g->cellList()) {
    int id = g->getData(c).nodeid;
    g->getData(c).emb = 0;
    maps[id-1] = c;
  }
  std::cout<<"after nets\n";
  std::ifstream f(filename.c_str());
  std::string line;
  std::stringstream ss(line);
  while (std::getline(f, line)) {
    int i = 0;
    std::stringstream ss(line);
    int val;
    ss >> val; //node id
    if (val >= nsize) break;
    GNode node = maps[val-1];
    int point;
    ss >> point; 
    g->getData(node).emb = point;
    
  }
  
  f.close();

  
  std::cout<<"end of norm\n";
}
