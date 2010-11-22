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

typedef Galois::Graph::FirstGraph<Element,void,false>            Graph;
typedef Galois::Graph::FirstGraph<Element,void,false>::GraphNode GNode;


#include "Subgraph.h"
#include "Mesh.h"
#include "Cavity.h"

#include "Support/ThreadSafe/simple_lock.h"
#include "Support/ThreadSafe/TSQueue.h"

#ifdef WITH_VTUNE
#include "/opt/intel/vtune_amplifier_xe_2011/include/libittnotify.h"
#endif

Graph* mesh;
int threads = 1;

void process(GNode item, Galois::Context<GNode>& lwl) {
  if (!mesh->containsNode(item))
    return;

  item.getData(Galois::Graph::ALL, lwl.getRuntimeContext()); //lock

  Cavity cav(mesh);
  cav.initialize(item, &lwl);
  cav.build(&lwl);
  cav.update(&lwl);

  for (Subgraph::iterator ii = cav.getPre().begin(),
	 ee = cav.getPre().end(); ii != ee; ++ii) 
    mesh->removeNode(*ii, Galois::Graph::NONE, lwl.getRuntimeContext());

  //add new data
  for (Subgraph::iterator ii = cav.getPost().begin(),
	 ee = cav.getPost().end(); ii != ee; ++ii) {
    GNode node = *ii;
    mesh->addNode(node, Galois::Graph::ALL, lwl.getRuntimeContext());
    Element& element = node.getData(Galois::Graph::ALL, lwl.getRuntimeContext());
    if (element.isBad()) {
      lwl.push(node);
    }
  }

  for (Subgraph::edge_iterator ii = cav.getPost().edge_begin(),
	 ee = cav.getPost().edge_end(); ii != ee; ++ii) {
    Subgraph::tmpEdge edge = *ii;
    //bool ret = 
    mesh->addEdge(edge.src, edge.dst, Galois::Graph::ALL, lwl.getRuntimeContext()); //, edge.data);
    //assert ret;
  }
  if (mesh->containsNode(item)) {
    lwl.push(item);
  }
}

template<typename WLTY>
void refine(Mesh& m, WLTY& wl) {
  //  if (threads == 1) {
  //    while (wl.size()) {
  //      bool suc;
  //      GNode N = wl.pop(suc);
  //      process(N, wl);
  //    }
  //  } else {
  //Galois::setMaxThreads(threads);
  using namespace Galois::Scheduling;
#ifdef WITH_VTUNE
  __itt_resume();
#endif
  Galois::for_each(wl, process, 
		   Priority.first<ChunkedFIFO>().thenLocally<LIFO>());
    //  }
#ifdef WITH_VTUNE
  __itt_pause();
#endif
}


using namespace std;

int main(int argc, char** argv) {

  if (argc < 2) {
    cout << "Arguments: [-t threads] <input file>\n";
    return 1;
  }

  int inputFileAt = 1;
  if (std::string("-t") == argv[1]) {
    inputFileAt = 3;
    threads = atoi(argv[2]);
  }

  cout << "\nLonestar Benchmark Suite v3.0\n"
       << "Copyright (C) 2007, 2008, 2009, 2010 The University of Texas at Austin\n"
       << "http://iss.ices.utexas.edu/lonestar/\n"
       << "\n"
       << "application: Delaunay Mesh Refinement (c++ version)\n"
       << "Refines a Delaunay triangulation mesh such that no angle\n"
       << "in the mesh is less than 30 degrees\n"
       << "http://iss.ices.utexas.edu/lonestar/delaunayrefinement.html\n"
       << "\n";

  mesh = new Graph();
  Mesh m;
  m.read(mesh, argv[inputFileAt]);
  //  threadsafe::ts_queue<GNode> wl;
  std::vector<GNode> wl;
  int numbad = m.getBad(mesh, wl);

  cout << "configuration: " << mesh->size() << " total triangles, " << numbad << " bad triangles\n"
       << "number of threads: " << threads << "\n"
       << "\n";

  Galois::setMaxThreads(threads);
  Galois::Launcher::startTiming();
  refine(m, wl);
  Galois::Launcher::stopTiming();
  
  cout << "STAT: Time " << Galois::Launcher::elapsedTime() << "\n";

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

