#include "Galois/Galois.h"
#include "Galois/Graphs/Bag.h"
#include "Lonestar/BoilerPlate.h"

#include <boost/iterator/counting_iterator.hpp>

typedef Galois::Graph::Bag<int>::pointer IntPtrs;

struct InsertBody : public Galois::Runtime::Lockable {
  IntPtrs pBodies;
  InsertBody() { }
  InsertBody(IntPtrs& b): pBodies(b) { }
  template<typename Context>
  void operator()(int in, const Context& cnx) {
    Galois::Runtime::LL::gPrint("i: ", in, "\n");
    pBodies->push(in);
  }
  // serialization functions
  typedef int tt_has_serialize;
  void serialize(Galois::Runtime::SerializeBuffer& s) const {
    gSerialize(s,pBodies);
  }
  void deserialize(Galois::Runtime::DeSerializeBuffer& s) {
    gDeserialize(s,pBodies);
  }
};

struct PrintInt : public Galois::Runtime::Lockable {
  typedef int tt_does_not_need_stats;
  PrintInt() { }

  template<typename Context>
  void operator()(int i, Context& cnx) {
    Galois::Runtime::LL::gPrint("i: ", i, " host: ", Galois::Runtime::NetworkInterface::ID, "\n");
    //    std::stringstream ss;
    //    ss << "i: " << i << " host: " << Galois::Runtime::Distributed::networkHostID << "\n";
    //    std::cout << ss.str();
  }

  // serialization functions
  typedef int tt_has_serialize;
  void serialize(Galois::Runtime::SerializeBuffer& s) const { }
  void deserialize(Galois::Runtime::DeSerializeBuffer& s) { }
};

int main(int argc, char** argv) {
  LonestarStart(argc, argv, nullptr, nullptr, nullptr);

  // check the host id and initialise the network
  Galois::Runtime::getSystemNetworkInterface().start();

  IntPtrs pBodies = Galois::Graph::Bag<int>::allocate();
  Galois::for_each<>(boost::counting_iterator<unsigned>(0), boost::counting_iterator<unsigned>(10), InsertBody(pBodies));

  Galois::for_each_local<>(pBodies,PrintInt());

  // master_terminate();
  Galois::Runtime::getSystemNetworkInterface().terminate();

  return 0;
}
