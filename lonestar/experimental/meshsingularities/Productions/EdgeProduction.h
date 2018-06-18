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

class EdgeProduction : public AbstractProduction {

private:
  const int bOffset;
  const int cOffset;

  Vertex* recursiveGraphGeneration(int low_range, int high_range,
                                   GraphNode mergingDstNode,
                                   GraphNode bsSrcNode);
  void generateGraph();

public:
  EdgeProduction(std::vector<int>* productionParameters,
                 std::vector<EquationSystem*>* inputData)
      : AbstractProduction(productionParameters, inputData), bOffset(2),
        cOffset(1) {
    this->inputData = inputData;
    generateGraph();
  };
  virtual std::vector<double>* getResult();
  virtual void Execute(EProduction productionToExecute, Vertex* v,
                       EquationSystem* input);
  virtual Vertex* getRootVertex();
  virtual Graph* getGraph();
  void B(Vertex* v, EquationSystem* inData) const;
  void BSB(Vertex* v, EquationSystem* inData) const;
  void C(Vertex* v, EquationSystem* inData) const;
  void BSC(Vertex* v, EquationSystem* inData) const;
  void D(Vertex* v, EquationSystem* inData) const;
  void BSD(Vertex* v, EquationSystem* inData) const;
  void MB(Vertex* v) const;
  void BSMB(Vertex* v) const;
  void MC(Vertex* v) const;
  void BSMC(Vertex* v) const;
  void MD(Vertex* v) const;
  void BSMD(Vertex* v) const;
  void MBLeaf(Vertex* v) const;
  void BSMBLeaf(Vertex* v) const;
  void MBC(Vertex* v, bool root) const;
  void BSMBC(Vertex* v) const;
  void Copy(Vertex* v, EquationSystem* inData) const;
};

#endif /* EDGEPRODUCTION_H_ */
