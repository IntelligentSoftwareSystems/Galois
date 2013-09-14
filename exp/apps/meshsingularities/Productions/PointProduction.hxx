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

	void generateGraph();

	virtual void recursiveGraphGeneration(int low_range, int high_range,
			GraphNode bsSrcNode,
			GraphNode mergingDstNode,
			Vertex *parent);

	virtual GraphNode addNode(int incomingEdges,
			int outgoingEdges,
			int leafNumber,
			EProduction production,
			GraphNode src,
			GraphNode dst,
			Vertex *v,
			EquationSystem *system);

  public:
	PointProduction(std::vector<int>* productionParameters,
					std::vector<EquationSystem*> *inputData) : AbstractProduction(productionParameters, inputData),
		interfaceSize((*productionParameters)[0]),
		leafSize((*productionParameters)[1]),
		a1Size((*productionParameters)[2]),
		anSize((*productionParameters)[3]),
		offset((*productionParameters)[1] - 2*(*productionParameters)[0]),
		a1Offset((*productionParameters)[2] - (*productionParameters)[1]),
		anOffset((*productionParameters)[3] - (*productionParameters)[1]) {
		generateGraph();
  	  };

	virtual void Execute(EProduction productionToExecute, Vertex* v, EquationSystem* input);
	virtual Vertex *getRootVertex();
	virtual Graph *getGraph();
    void A1(Vertex *v, EquationSystem *inData) const;
    void A(Vertex *v, EquationSystem *inData) const;
    void AN(Vertex *v, EquationSystem *inData) const;
    void A2Node(Vertex *v) const;
    void A2Root(Vertex *v) const;
    void BS(Vertex *v) const;


    int getInterfaceSize() const;
    int getLeafSize() const;
    int getA1Size() const;
    int getANSize() const;

};

#endif /* POINTPRODUCTION_HXX_ */
