#include "Galois/Galois.h"
#include "Galois/Graphs/Graph3.h"
#include "Lonestar/BoilerPlate.h"

#include <boost/iterator/counting_iterator.hpp>

#include <iostream>

using namespace Galois::Graph;

typedef ThirdGraph<int,int,EdgeDirection::Out> G;
typedef typename G::pointer Gptr;

int num = 1000;

bool check(unsigned x, Gptr Gr, unsigned& r) {
  auto b = Gr->begin();
  std::advance(b, x);
  r = std::distance(b, Gr->end());
  return r == (num - x);
}

struct op {
  Gptr Gr;

  op(const Gptr& p) : Gr(p) {}
  op() {}

  template<typename Context>
  void operator()(unsigned x, Context& cnx) const {
    static Galois::Runtime::LL::SimpleLock L;
    unsigned r;
    if (!check(x, Gr, r)) {
      L.lock();
      std::cout << "Mismatch " << x << " " << num - x << " " << r << "\n";
      L.unlock();
    }
  }
};

struct cop {
  Gptr Gr;

  cop(const Gptr& p) : Gr(p) {}
  cop() {}

  template<typename Context>
  void operator()(unsigned x, Context& cnx) const {
    G::NodeHandle node = Gr->createNode(x);
    Gr->addNode(node);
    // static Galois::Runtime::LL::SimpleLock<true> L;
    // L.lock();
    //std::cout << x << " " << Galois::Runtime::networkHostID << " " << Galois::Runtime::LL::getTID() << "\n";
    // L.unlock();
  }
};

int main(int argc, char** argv) {

  LonestarStart(argc, argv, nullptr, nullptr, nullptr);

  // check the host id and initialise the network
  Galois::Runtime::getSystemNetworkInterface().start();

  Gptr Gr = G::allocate();

  Galois::for_each<>(boost::counting_iterator<unsigned>(0), boost::counting_iterator<unsigned>(num), cop(Gr));
  std::cout << "\nDone Building\n";

  std::cout << "\nTotal Size " << std::distance(Gr->begin(), Gr->end()) << " (should be " << num << ")\n";
  unsigned r;
  for (int x = 0; x < num; ++x)
    if (!check(x, Gr, r))
      std::cout << "Mismatch " << x << " " << num - x << " " << r << "\n";
  std::cout << "\nDone Local Check\n";
  Galois::for_each<>(boost::counting_iterator<unsigned>(0), boost::counting_iterator<unsigned>(num), op(Gr));

  std::cout << "\nDone Remote Check\n";

  // master_terminate();
  Galois::Runtime::getSystemNetworkInterface().terminate();

  return 0;
}

