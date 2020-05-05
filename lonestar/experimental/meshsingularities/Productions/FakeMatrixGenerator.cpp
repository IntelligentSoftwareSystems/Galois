/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
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
 * FakeMatrixGenerator.cpp
 *
 *  Created on: Aug 28, 2013
 *      Author: kjopek
 */

#include "FakeMatrixGenerator.h"

std::vector<EquationSystem*>*
FakeMatrixGenerator::CreateMatrixAndRhs(TaskDescription& taskDescription) {

  std::vector<EquationSystem*>* leafVector = new std::vector<EquationSystem*>();

  const int iSize    = this->getiSize(taskDescription.polynomialDegree);
  const int leafSize = this->getLeafSize(taskDescription.polynomialDegree);
  const int a1Size   = this->getA1Size(taskDescription.polynomialDegree);
  const int aNSize   = this->getANSize(taskDescription.polynomialDegree);

  for (int i = 0; i < taskDescription.nrOfTiers; ++i) {
    int n;
    if (i == 0) {
      n = a1Size;
    } else if (i == taskDescription.nrOfTiers - 1) {
      n = aNSize;
    } else {
      n = leafSize + 2 * iSize;
    }

    EquationSystem* system = new EquationSystem(n);
    for (int j = 0; j < n; ++j) {
      for (int k = 0; k < n; ++k) {
        system->matrix[j][k] = (INT_MAX - rand()) / INT_MAX * 1.0;
      }
      system->rhs[j] = 1.0;
    }

    leafVector->push_back(system);
  }

  return leafVector;
}

void FakeMatrixGenerator::checkSolution(std::map<int, double>* solution_map,
                                        double (*f)(int dim, ...)) {
  // empty
}
