#include "Galois/Galois.h"
#include "Galois/Graphs/Bag.h"
#include "Lonestar/BoilerPlate.h"

#include <boost/iterator/counting_iterator.hpp>

typedef galois::graphs::Bag<int>::pointer IntPtrs;

struct InsertBody { 
  IntPtrs pBodies;

  template<typename Context>
  void operator()(int i, const Context& ctx) {
    galois::runtime::LL::gPrint("host: ", galois::runtime::NetworkInterface::ID, " pushing: ", i, "\n");
    pBodies->push(i);
  }

  //Trivially_copyable
  typedef int tt_is_copyable;
};

struct PrintInt {
  template<typename Context>
  void operator()(int i, Context& ctx) {
    galois::runtime::LL::gPrint("host: ", galois::runtime::NetworkInterface::ID, " received: ", i, "\n");
  }
};

int main(int argc, char** argv) {
  LonestarStart(argc, argv, nullptr, nullptr, nullptr);
  galois::runtime::getSystemNetworkInterface().start();

  IntPtrs pBodies = galois::graphs::Bag<int>::allocate();
  galois::for_each(boost::counting_iterator<int>(0), boost::counting_iterator<int>(10), InsertBody { pBodies });
  galois::for_each_local(pBodies, PrintInt());

  galois::runtime::getSystemNetworkInterface().terminate();

  return 0;
}
