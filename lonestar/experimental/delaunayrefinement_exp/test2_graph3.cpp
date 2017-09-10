#include "Galois/Galois.h"
#include "Galois/ParallelSTL/ParallelSTL.h"
#include "Galois/Graphs/Bag.h"
#include "Galois/Statistic.h"

#include "llvm/Support/CommandLine.h"
#include "Lonestar/BoilerPlate.h"

#include "Galois/WorkList/WorkListAlt.h"
#include "Galois/WorkList/WorkListDebug.h"

#include "Galois/Graphs/Graph3.h"
#include "Galois/Runtime/DistSupport.h"

#include <cassert>
#include <stdlib.h>

#include "Edge.h"
#include "Galois/Runtime/Serialize.h"

#include "Galois/Runtime/Context.h"

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
typedef Galois::Graph::ThirdGraph<int, void, Galois::Graph::EdgeDirection::Un> Graph;
typedef Graph::NodeHandle GNode;
typedef typename Graph::pointer Graphp;



namespace cll = llvm::cl;

static const char* name = "Delaunay Mesh Refinement";
static const char* desc = "Refines a Delaunay triangulation mesh such that no angle in the mesh is less than 30 degrees\n";
static const char* url = "delaunay_mesh_refinement";

static cll::opt<std::string> filename(cll::Positional, cll::desc("<input file>"), cll::Required);



// dummy for_each
/*
struct initialize {

  Graphp g;
  void static go(Graphp g, std::vector<int>& elements) {
    Galois::for_each_local(elements.begin(), elements.end(), initialize{g}, Galois::loopname("initialize"));
  }

  void operator() (node_data& item, Galois::UserContext<node_data>& cnx) {
    GNode n = g->createNode(item);
    g->addNode(n);
  }

  typedef int tt_is_copyable;
};

*/

struct initialize : public Galois::Runtime::Lockable {
  Graphp g;
  initialize() {}
  initialize(Graphp _g): g(_g) {}

  void operator()(int& item, Galois::UserContext<int>& cnx) {
    GNode n = g->createNode(item);
    g->addNode(n);
  }
 // serialization functions
  typedef int tt_has_serialize;
  void serialize(Galois::Runtime::SerializeBuffer& s) const {
    gSerialize(s,g);
  }
  void deserialize(Galois::Runtime::DeSerializeBuffer& s) {
    gDeserialize(s,g);
  }



};

/*
struct Point;
typedef Galois::Runtime::PerThreadDist<Point> DistPoint;
typedef Galois::Runtime::gptr<Point> PtrPoint;


struct Point: Galois::Runtime::Lockable {

  typedef int tt_is_copyable;

  using Base = Galois::Runtime::Lockable;

  int x;
  int y;

  Point (): Base (),  x(11), y(22) {}

  Point (int _x, int _y): Base (), x (_x), y (_y)
  {}

  Point (DistPoint): Base (), x(10), y(20) {}

  Point (DistPoint, Galois::Runtime::DeSerializeBuffer& buf): Base (), x(100), y(200) {}

  void getInitData (Galois::Runtime::SerializeBuffer& buf) {

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
        Galois::Runtime::NetworkInterface::ID, tid, p->x, p->y);
  }
};
*/

int main(int argc, char** argv) {

  LonestarStart(argc, argv, name, desc, url);
  Galois::StatManager statManager;

  Galois::Runtime::getSystemNetworkInterface().start();

/*
  DistPoint dp = DistPoint::allocate ();

  PtrPoint p = dp.local ();
  std::printf ("After allocate: Host=%u, Thread=%u, x=%d, y=%d\n",
      Galois::Runtime::NetworkInterface::ID, 0, p->x, p->y);

  Galois::on_each (OnEachFunc {dp}, "test-loop");
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

   Galois::for_each(vec_items.begin(), vec_items.end(),initialize(g), Galois::loopname("initializing"));//, Galois::wl<Galois::WorkList::StableIterator<>>());



  Galois::Runtime::getSystemNetworkInterface().terminate();
}
