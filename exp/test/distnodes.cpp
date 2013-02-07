#include "Galois/Galois.h"
#include "Galois/Statistic.h"
#include "Galois/Graphs/Graph3.h"
#include "Lonestar/BoilerPlate.h"

#include <iostream>

using namespace Galois::Graph;

template<typename nd, typename ed, EdgeDirection dir>
using G = ThirdGraph<nd,ed,dir>;


struct op {
  template<typename T, typename Context>
  void operator()(const T& node, const Context& cnx) {
    node->createEdge(node, -node->getData());
  }
};

int main(int argc, char** argv) {
  Galois::StatManager statManager;
  LonestarStart(argc, argv, nullptr, nullptr, nullptr);

  // check the host id and initialise the network
  Galois::Runtime::Distributed::networkStart();

  typedef G<int, int , EdgeDirection::Out> GTy;
  GTy Gr;

  for (int x = 0; x < 100; ++x)
    Gr.createNode(x);

  Galois::for_each<>(Gr.begin(), Gr.end(), op());

  for (auto ii = Gr.begin(), ee = Gr.end(); ii != ee; ++ii)
    std::cout << (*ii)->getData() << " " << std::distance((*ii)->begin(), (*ii)->end()) << " ";
  std::cout << "\n";

  // master_terminate();
  Galois::Runtime::Distributed::networkTerminate();

  ///debugging stuff below this point
  if (false) {
    Galois::Runtime::Distributed::SerializeBuffer B;
    //serialize the pointer
    auto oldB = *Gr.begin();
    gSerialize(B,oldB);
    //serialize the node
    gSerialize(B,*oldB);
    
    B.print(std::cout);
    cout << "\n";
    
    Galois::Runtime::Distributed::DeSerializeBuffer D(B.size());
    memcpy(D.linearData(), B.linearData(), B.size());
    
    D.print(std::cout);
    cout << "\n";
    
    // read the header first
    uintptr_t tmp;
    gDeserialize(D,tmp);
    
    typename GTy::NodeHandle foo;
    gDeserialize(D,foo);
    
    auto bar = Gr.createNode();
    gDeserialize(D,*bar);
    
    B.print(std::cout);
    cout << "\n";
    D.print(std::cout);
    cout << "\n";
    
    oldB.dump();
    std::cout << "\n";
    foo.dump();
    std::cout << "\n";
    (*oldB).dump(std::cout);
    std::cout << "\n";
    (*bar).dump(std::cout);
    std::cout << "\n";
  }
  return 0;
}
