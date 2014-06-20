#include "Galois/Galois.h"
#include "Galois/Graphs/Bag.h"
#include "Lonestar/BoilerPlate.h"

#include <boost/iterator/counting_iterator.hpp>

typedef Galois::Graph::Bag<int>::pointer IntPtrs;

struct InsertBody { 
  IntPtrs pBodies;

  template<typename Context>
  void operator()(int i, const Context& ctx) {
    Galois::Runtime::LL::gPrint("host: ", Galois::Runtime::NetworkInterface::ID, " pushing: ", i, "\n");
    pBodies->push(i);
  }

  //Trivially_copyable
  typedef int tt_is_copyable;
};

struct PrintInt {
  template<typename Context>
  void operator()(int i, Context& ctx) {
    Galois::Runtime::LL::gPrint("host: ", Galois::Runtime::NetworkInterface::ID, " received: ", i, "\n");
  }
};

int main(int argc, char** argv) {
  LonestarStart(argc, argv, nullptr, nullptr, nullptr);
  Galois::Runtime::getSystemNetworkInterface().start();

  IntPtrs pBodies = Galois::Graph::Bag<int>::allocate();
  Galois::for_each(boost::counting_iterator<int>(0), boost::counting_iterator<int>(10), InsertBody { pBodies });
  Galois::for_each_local(pBodies, PrintInt());

  Galois::Runtime::getSystemNetworkInterface().terminate();

  return 0;
}
