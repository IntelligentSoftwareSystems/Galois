#include "Galois/Galois.h"
#include "Galois/Graphs/Graph3.h"
#include "Lonestar/BoilerPlate.h"

#include <boost/iterator/counting_iterator.hpp>

#include <iostream>

using namespace Galois::Graph;

typedef ThirdGraph<int,int,EdgeDirection::Out> G;
typedef Galois::Runtime::Distributed::gptr<G> Gptr;

struct op {
  Gptr Gr;

  op(const Gptr& p) : Gr(p) {}
  op() {}

  template<typename Context>
  void operator()(unsigned x, Context& cnx) const {
    auto b = Gr->begin();
    std::advance(b, x);
    if (std::distance(b, Gr->end()) != 100 - x)
      std::cerr << "Mismatch\n";
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
    Gr->createNode(x);
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

  Gptr Gr(new G());
  
  Galois::for_each<>(boost::counting_iterator<unsigned>(0), boost::counting_iterator<unsigned>(100), cop(Gr));

  Galois::for_each<>(boost::counting_iterator<unsigned>(0), boost::counting_iterator<unsigned>(100), op(Gr));

  for (int x = 0; x < 100; ++x) {
    auto b = Gr->begin();
    std::advance(b, x);
    if (std::distance(b, Gr->end()) != 100 - x)
      std::cerr << "Mismatch\n";
  }

  // master_terminate();
  Galois::Runtime::Distributed::networkTerminate();

  return 0;
}

