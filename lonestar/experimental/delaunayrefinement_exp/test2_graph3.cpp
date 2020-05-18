/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
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
 */

#include "galois/Galois.h"
#include "galois/ParallelSTL/ParallelSTL.h"
#include "galois/graphs/Bag.h"
#include "galois/Timer.h"

#include "llvm/Support/CommandLine.h"
#include "Lonestar/BoilerPlate.h"

#include "galois/worklists/WorkListAlt.h"
#include "galois/worklists/WorkListDebug.h"

#include "galois/graphs/Graph3.h"
#include "galois/runtime/DistSupport.h"

#include <cassert>
#include <stdlib.h>

#include "Edge.h"
#include "galois/runtime/Serialize.h"

#include "galois/runtime/Context.h"

#include <vector>
#include <algorithm>
#include <iostream>
#include <string.h>
#include <cassert>

struct node_data {

  int data;

  typedef int tt_is_copyable;
};

// dummy graph3
typedef galois::graphs::ThirdGraph<int, void, galois::graphs::EdgeDirection::Un>
    Graph;
typedef Graph::NodeHandle GNode;
typedef typename Graph::pointer Graphp;

namespace cll = llvm::cl;

static const char* name = "Delaunay Mesh Refinement";
static const char* desc = "Refines a Delaunay triangulation mesh such that no "
                          "angle in the mesh is less than 30 degrees\n";
static const char* url = "delaunay_mesh_refinement";

static cll::opt<std::string> filename(cll::Positional,
                                      cll::desc("<input file>"), cll::Required);

// dummy for_each
/*
struct initialize {

  Graphp g;
  void static go(Graphp g, std::vector<int>& elements) {
    galois::for_each(elements.begin(), elements.end(), initialize{g},
galois::loopname("initialize"));
  }

  void operator() (node_data& item, galois::UserContext<node_data>& cnx) {
    GNode n = g->createNode(item);
    g->addNode(n);
  }

  typedef int tt_is_copyable;
};

*/

struct initialize : public galois::runtime::Lockable {
  Graphp g;
  initialize() {}
  initialize(Graphp _g) : g(_g) {}

  void operator()(int& item, galois::UserContext<int>& cnx) {
    GNode n = g->createNode(item);
    g->addNode(n);
  }
  // serialization functions
  typedef int tt_has_serialize;
  void serialize(galois::runtime::SerializeBuffer& s) const {
    gSerialize(s, g);
  }
  void deserialize(galois::runtime::DeSerializeBuffer& s) {
    gDeserialize(s, g);
  }
};

/*
struct Point;
typedef galois::runtime::PerThreadDist<Point> DistPoint;
typedef galois::runtime::gptr<Point> PtrPoint;


struct Point: galois::runtime::Lockable {

  typedef int tt_is_copyable;

  using Base = galois::runtime::Lockable;

  int x;
  int y;

  Point (): Base (),  x(11), y(22) {}

  Point (int _x, int _y): Base (), x (_x), y (_y)
  {}

  Point (DistPoint): Base (), x(10), y(20) {}

  Point (DistPoint, galois::runtime::DeSerializeBuffer& buf): Base (), x(100),
y(200) {}

  void getInitData (galois::runtime::SerializeBuffer& buf) {

  }
};


struct OnEachFunc {

  DistPoint dp;

  OnEachFunc (void):dp(){}

  OnEachFunc (DistPoint _dp): dp (_dp) {}

  typedef int tt_is_copyable;

  void operator () (unsigned tid, unsigned numT) {
    //assert (dp != nullptr);

    PtrPoint p = dp.local();

    std::printf ("In on_each: Host=%u, Thread=%u, x=%d, y=%d\n",
        galois::runtime::NetworkInterface::ID, tid, p->x, p->y);
  }
};
*/

int main(int argc, char** argv) {

  LonestarStart(argc, argv, name, desc, url);
  galois::StatManager statManager;

  galois::runtime::getSystemNetworkInterface().start();

  /*
    DistPoint dp = DistPoint::allocate ();

    PtrPoint p = dp.local ();
    std::printf ("After allocate: Host=%u, Thread=%u, x=%d, y=%d\n",
        galois::runtime::NetworkInterface::ID, 0, p->x, p->y);

    galois::on_each (OnEachFunc {dp}, "test-loop");
  */

  Graphp g;
  g = Graph::allocate();

  std::vector<int> vec_items;
  for (int i = 0; i < 100; i++) {
    vec_items.push_back(i);
  }
  // for (auto ii : vec_items) {
  // std::cout << ii.data << "\n";
  //}

  galois::for_each(
      vec_items.begin(), vec_items.end(), initialize(g),
      galois::loopname(
          "initializing")); //,
                            // galois::wl<galois::worklists::StableIterator<>>());

  galois::runtime::getSystemNetworkInterface().terminate();
}
