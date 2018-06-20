/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 */

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
