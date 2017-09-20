#include "Galois/Graphs/Graph3.h"

#include <iostream>

using namespace galois::Graph;

template<typename nd, typename ed, EdgeDirection dir>
using G = ThirdGraph<nd,ed,dir>;

G< int,  int, EdgeDirection::Out> Giio;
G< int, void, EdgeDirection::Out> Givo;
G<void,  int, EdgeDirection::Out> Gvio;
G<void, void, EdgeDirection::Out> Gvvo;
G< int,  int, EdgeDirection::InOut> Giii;
G< int, void, EdgeDirection::InOut> Givi;
G<void,  int, EdgeDirection::InOut> Gvii;
G<void, void, EdgeDirection::InOut> Gvvi;
G< int,  int, EdgeDirection::Un> Giiu;
G< int, void, EdgeDirection::Un> Givu;
G<void,  int, EdgeDirection::Un> Gviu;
G<void, void, EdgeDirection::Un> Gvvu;

int main(int argc, const char** argv) {
  int i = 4;

  //Directed out edge
  std::cout << Giio.createNode(2)->getData() << " "
	    << Giio.createNode(i)->getData() << " "
	    << Givo.createNode(2)->getData() << " "
	    << Givo.createNode(i)->getData() << "\n";
  
  Gvio.createNode();
  Gvvo.createNode();

  Giio.createNode()->createEdge(Giio.createNode(), 3);
  Giio.createNode()->createEdge(Giio.createNode(), i);
  Gvio.createNode()->createEdge(Gvio.createNode(), 3);
  Gvio.createNode()->createEdge(Gvio.createNode(), i);
  Givo.createNode()->createEdge(Givo.createNode());
  Givo.createNode()->createEdge(Givo.createNode());
  Gvvo.createNode()->createEdge(Gvvo.createNode());
  Gvvo.createNode()->createEdge(Gvvo.createNode());
  
  //Directed in out edge
  std::cout << Giii.createNode(2)->getData() << " "
	    << Giii.createNode(i)->getData() << " "
	    << Givi.createNode(2)->getData() << " "
	    << Givi.createNode(i)->getData() << "\n";

  Gvii.createNode();
  Gvvi.createNode();

  //Undirected
  std::cout << Giiu.createNode(2)->getData() << " "
	    << Giiu.createNode(i)->getData() << " "
	    << Givu.createNode(2)->getData() << " "
	    << Givu.createNode(i)->getData() << "\n";

  Gviu.createNode();
  Gvvu.createNode();

  return 0;
}
