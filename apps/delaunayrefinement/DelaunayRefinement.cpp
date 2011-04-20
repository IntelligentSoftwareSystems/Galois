/* 
 
   Lonestar DelaunayRefinement: Refinement of an initial, unrefined Delaunay
   mesh to eliminate triangles with angles < 30 degrees, using a
   variation of Chew's algorithm.
 
   Authors: Milind Kulkarni 

   Copyright (C) 2007, 2008 The University of Texas at Austin
 
   Licensed under the Eclipse Public License, Version 1.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
 
   http://www.eclipse.org/legal/epl-v10.html
 
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
 
   File: DelaunayRefinement.cpp
 
   Created: February 5th, 2008 by Milind Kulkarni (initial version)
 
*/ 

#include <iostream>
#include <stack>
#include <sys/time.h>
#include <limits.h>
#include <math.h>
#include <string.h>
#include <cassert>

#include "Element.h"

#include "Galois/Launcher.h"
#include "Galois/Graphs/Graph.h"
#include "Galois/Galois.h"

#include "Lonestar/Banner.h"
#include "Lonestar/CommandLine.h"

static const char* name = "Delaunay Mesh Refinement";
static const char* description = "Refines a Delaunay triangulation mesh such that no angle in the mesh is less than 30 degrees\n";
static const char* url = "http://iss.ices.utexas.edu/lonestar/delaunayrefinement.html";
static const char* help = "<input file>";

typedef Galois::Graph::FirstGraph<Element,void,false>            Graph;
typedef Galois::Graph::FirstGraph<Element,void,false>::GraphNode GNode;


#include "Subgraph.h"
#include "Mesh.h"
#include "Cavity.h"

Graph* mesh;
int threads = 1;

struct process {
  template<typename Context>
  void operator()(GNode item, Context& lwl) {
    if (!mesh->containsNode(item))
      return;
    
    item.getData(Galois::Graph::ALL); //lock

    Cavity cav(mesh, &lwl);
    cav.initialize(item);
    cav.build();
    cav.update();
    
    for (Subgraph::iterator ii = cav.getPre().begin(),
	   ee = cav.getPre().end(); ii != ee; ++ii) 
      mesh->removeNode(*ii, Galois::Graph::NONE);
    
    //add new data
    for (Subgraph::iterator ii = cav.getPost().begin(),
	   ee = cav.getPost().end(); ii != ee; ++ii) {
      GNode node = *ii;
      mesh->addNode(node, Galois::Graph::ALL);
      Element& element = node.getData(Galois::Graph::ALL);
      if (element.isBad()) {
	lwl.push(node);
      }
    }
    
    for (Subgraph::edge_iterator ii = cav.getPost().edge_begin(),
	   ee = cav.getPost().edge_end(); ii != ee; ++ii) {
      Subgraph::tmpEdge edge = *ii;
      //bool ret = 
      mesh->addEdge(edge.src, edge.dst, Galois::Graph::ALL); //, edge.data);
      //assert ret;
    }
    if (mesh->containsNode(item)) {
      lwl.push(item);
    }
  }
};

template<typename WLTY>
void refine(Mesh& m, WLTY& wl) {
  Galois::for_each(wl, process());
}


using namespace std;

int main(int argc, const char** argv) {

  std::vector<const char*> args = parse_command_line(argc, argv, help);

  if (args.size() != 1) {
    std::cout << "incorrect number of arguments, use -help for usage information\n";
    return 1;
  }
  printBanner(std::cout, name, description, url);

  mesh = new Graph();
  Mesh m;
  m.read(mesh, args[0]);
  std::vector<GNode> wl;
  int numbad = m.getBad(mesh, wl);

  cout << "configuration: " << mesh->size() << " total triangles, " << numbad << " bad triangles\n"
       << "number of threads: " << threads << "\n"
       << "\n";

  using namespace GaloisRuntime::WorkList;
  LocalQueues<GNode, ChunkedBag<GNode, 1024>, LIFO<GNode> > wl2;
  wl2.fill_initial(wl.begin(), wl.end());

  Galois::Launcher::startTiming();
  refine(m, wl2);
  Galois::Launcher::stopTiming();
  
  cout << "STAT: Time " << Galois::Launcher::elapsedTime() << "\n";

  if (!skipVerify) {
    if (!m.verify(mesh)) {
      cerr << "Refinement failed.\n";
      assert(0 && "Refinement failed");
      abort();
    }
    
    int size = m.getBad(mesh, wl);
    if (size != 0) {
      cerr << "Refinement failed with " << size << " remaining triangles.\n";
      assert(0 && "Refinement failed");
      abort();
    }
    cout << "Refinement OK\n";
  }
  return 0;
}

