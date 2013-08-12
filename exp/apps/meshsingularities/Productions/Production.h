#ifndef PRODUCTION_H
#define PRODUCTION_H

#include "Vertex.h"

class AbstractProduction {

  private:
	int interfaceSize;
	int leafSize;
	int a1Size;
	int anSize;

  public:
	AbstractProduction(int iSize, int lSize, int a1Size, int anSize) :
		interfaceSize(iSize), leafSize(lSize),
		a1Size(a1Size), anSize(anSize) {};

    void A1(Vertex *v, EquationSystem *inData);
    void A(Vertex *v, EquationSystem *inData);
    void AN(Vertex *v, EquationSystem *inData);
    void A2(Vertex *v);
    void E(Vertex *v);
    void ERoot(Vertex *v);
    void BS(Vertex *v);

    int getInterfaceSize();
    int getLeafSize();
    int getA1Size();
    int getANSize();
};

class MES2DP2 : public AbstractProduction {
  public:
	MES2DP2() : AbstractProduction(5, 17, 21, 21) {} ;
};

class MES3DP2 : public AbstractProduction {
  public:
	MES3DP2() : AbstractProduction(19, 75, 117, 83) {} ;
};

#endif
