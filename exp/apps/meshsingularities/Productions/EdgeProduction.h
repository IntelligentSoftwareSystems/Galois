/*
 * EdgeProduction.h
 *
 *  Created on: Sep 4, 2013
 *      Author: dgoik
 */

#ifndef EDGEPRODUCTION_H_
#define EDGEPRODUCTION_H_

#include "Vertex.h"
#include "Production.h"

class EdgeProduction : public AbstractProduction{

  private:
	//const int bSize;
	//const int cSize;
	/*const int bInterfaceSize;
	const int cInterfaceSize;
	const int c2cCommonInterfaceSize;
	const int b2bCommonInterfaceSize;
	const int b2cCommonINterfaceSize;
	*/
	const int bOffset;
	const int cOffset;

  public:
	EdgeProduction(std::vector<int>* production_parameters) : AbstractProduction(production_parameters),
		bOffset(2), cOffset(1)
		{};

	virtual void Execute(EProduction productionToExecute, Vertex* v, EquationSystem* input);
    void B(Vertex *v, EquationSystem *inData) const;
    void C(Vertex *v, EquationSystem *inData) const;
    void D(Vertex *v, EquationSystem *inData) const;
    void MB(Vertex *v) const;
    void MC(Vertex *v) const;
    void MD(Vertex *v) const;
    void MBLeaf(Vertex *v) const;
    void MBC(Vertex *v, bool root) const;
    //void MBCRoot(Vertex *v) const;
    void Copy(Vertex* v, EquationSystem* inData) const;
    //void BS(Vertex *v) const;

};

#endif /* EDGEPRODUCTION_H_ */
