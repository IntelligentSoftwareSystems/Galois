#ifndef PRODUCTION_H
#define PRODUCTION_H

#include "Vertex.h"
#include "EProduction.hxx"
#include <vector>
class AbstractProduction {

  public:
	AbstractProduction(std::vector<int>* production_parameters)
  	{

  	};

	virtual ~AbstractProduction()
	{

	}
	virtual void Execute(EProduction productionToExecute, Vertex* v, EquationSystem* input) = 0;
  /*  void A1(Vertex *v, EquationSystem *inData) const;
    void A(Vertex *v, EquationSystem *inData) const;
    void AN(Vertex *v, EquationSystem *inData) const;
 //   void A2(Vertex *v) const;
    void A2Node(Vertex *v) const;
    void A2Root(Vertex *v) const;
//    void E(Vertex *v) const;
//    void ERoot(Vertex *v) const;
    void BS(Vertex *v) const;


    int getInterfaceSize() const;
    int getLeafSize() const;
    int getA1Size() const;
    int getANSize() const;*/

};

#endif
