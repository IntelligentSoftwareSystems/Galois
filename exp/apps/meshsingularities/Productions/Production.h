#ifndef PRODUCTION_H
#define PRODUCTION_H

#include "Vertex.h"

class AbstractProduction {

  private:
	const int interfaceSize;
	const int leafSize;
	const int a1Size;
	const int anSize;
	const int offset;
	const int a1Offset;
	const int anOffset;

  public:
	AbstractProduction(int iSize, int lSize, int a1Size, int anSize) :
		interfaceSize(iSize), leafSize(lSize),
		a1Size(a1Size), anSize(anSize),
		offset(lSize - 2*iSize),
		a1Offset(a1Size - lSize),
		anOffset(anSize - lSize) {};

    void A1(Vertex *v, EquationSystem *inData) const;
    void A(Vertex *v, EquationSystem *inData) const;
    void AN(Vertex *v, EquationSystem *inData) const;
    void A2(Vertex *v) const;
    void E(Vertex *v) const;
    void ERoot(Vertex *v) const;
    void BS(Vertex *v) const;

    int getInterfaceSize() const;
    int getLeafSize() const;
    int getA1Size() const;
    int getANSize() const;
};

#endif
