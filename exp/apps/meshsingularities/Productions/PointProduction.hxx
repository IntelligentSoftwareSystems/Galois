/*
 * PointProduction.hxx
 *
 *  Created on: Sep 12, 2013
 *      Author: dgoik
 */

#ifndef POINTPRODUCTION_HXX_
#define POINTPRODUCTION_HXX_

#include "Production.h"

class PointProduction : public AbstractProduction{

  private:
	const int interfaceSize;
	const int leafSize;
	const int a1Size;
	const int anSize;
	const int offset;
	const int a1Offset;
	const int anOffset;

  public:
	PointProduction(std::vector<int>* production_parameters) : AbstractProduction(production_parameters), interfaceSize((*production_parameters)[0]), leafSize((*production_parameters)[1]),
	a1Size((*production_parameters)[2]), anSize((*production_parameters)[3]), offset((*production_parameters)[1] - 2*(*production_parameters)[0]),
	a1Offset((*production_parameters)[2] - (*production_parameters)[1]), anOffset((*production_parameters)[3] - (*production_parameters)[1])
  	  {

  	  };

	virtual void Execute(EProduction productionToExecute, Vertex* v, EquationSystem* input);
    void A1(Vertex *v, EquationSystem *inData) const;
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
    int getANSize() const;

};

#endif /* POINTPRODUCTION_HXX_ */
