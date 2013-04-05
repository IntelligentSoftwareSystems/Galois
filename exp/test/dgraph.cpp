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
    static Galois::Runtime::LL::SimpleLock<true> L;
    unsigned r;
    if (!check(x, Gr, r)) {
      L.lock();
      std::cout << "Mismatch " << x << " " << num - x << " " << r << "\n";
      L.unlock();
    }
  }

  // serialization functions
  typedef int tt_has_serialize;
  void serialize(Galois::Runtime::Distributed::SerializeBuffer& s) const {
    gSerialize(s,Gr);
  }
  void deserialize(Galois::Runtime::Distributed::DeSerializeBuffer& s) {
    gDeserialize(s,Gr);
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
    // std::cout << x << " " << Galois::Runtime::Distributed::networkHostID << " " << Galois::Runtime::LL::getTID() << "\n";
    // L.unlock();
  }

  // serialization functions
  typedef int tt_has_serialize;
  void serialize(Galois::Runtime::Distributed::SerializeBuffer& s) const {
    gSerialize(s,Gr);
  }
  void deserialize(Galois::Runtime::Distributed::DeSerializeBuffer& s) {
    gDeserialize(s,Gr);
  }

};

int main(int argc, char** argv) {

  LonestarStart(argc, argv, nullptr, nullptr, nullptr);

  // check the host id and initialise the network
  Galois::Runtime::Distributed::networkStart();

  Gptr Gr = G::allocate();

  Galois::for_each<>(boost::counting_iterator<unsigned>(0), boost::counting_iterator<unsigned>(num), cop(Gr));
  std::cout << "\nDone Building\n";

  //  volatile int br = 1;
  //  while (br) {}
  std::cout << "\nTotal Size " << std::distance(Gr->begin(), Gr->end()) << " (should be " << num << ")\n";
  unsigned r;
  for (int x = 0; x < num; ++x)
    if (!check(x, Gr, r))
      std::cout << "Mismatch " << x << " " << num - x << " " << r << "\n";
  std::cout << "\nDone Local Check\n";
  // br = 0;
  // while (br) {}
  Galois::for_each<>(boost::counting_iterator<unsigned>(0), boost::counting_iterator<unsigned>(num), op(Gr));

  std::cout << "\nDone Remote Check\n";

  // master_terminate();
  Galois::Runtime::Distributed::networkTerminate();

  return 0;
}

