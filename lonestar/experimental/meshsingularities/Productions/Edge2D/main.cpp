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

#include <stdio.h>
#include "MatrixGenerator.hxx"
#include <map>
#include <time.h>
#include <map>
#include "../EquationSystem.h"

using namespace D2Edge;

double test_function(int dim, ...) {
  double* data  = new double[dim];
  double result = 0;
  va_list args;

  va_start(args, dim);
  for (int i = 0; i < dim; ++i) {
    data[i] = va_arg(args, double);
  }
  va_end(args);

  if (dim == 2) {

    result = 1;
    // result = data[0]*data[1]+data[0]+data[0]*data[1]*data[0]*data[1] + 11;
  } else {
    result = -1;
  }

  delete[] data;
  return result;
}

int main(int argc, char** argv) {
  GenericMatrixGenerator* matrixGenerator = new MatrixGenerator();

  TaskDescription taskDescription;
  taskDescription.dimensions = 2;
  taskDescription.nrOfTiers  = 1;
  taskDescription.size       = 2;
  taskDescription.function   = test_function;
  taskDescription.x          = -1;
  taskDescription.y          = -1;
  std::vector<EquationSystem*>* tiers =
      matrixGenerator->CreateMatrixAndRhs(taskDescription);

  EquationSystem* globalSystem = new EquationSystem(
      matrixGenerator->GetMatrix(), matrixGenerator->GetRhs(),
      matrixGenerator->GetMatrixSize());

  // globalSystem->print();
  globalSystem->eliminate(matrixGenerator->GetMatrixSize());
  globalSystem->backwardSubstitute(matrixGenerator->GetMatrixSize() - 1);

  /*
  std::list<EquationSystem*>::iterator it = tiers->begin();
  for(; it != tiers->end(); ++it)
  {
      (*it)->print();
      printf("--------------------\n");
  }
   */
  std::map<int, double>* result_map = new std::map<int, double>();
  for (int i = 0; i < matrixGenerator->GetMatrixSize(); i++) {
    (*result_map)[i] = globalSystem->rhs[i];
  }

  matrixGenerator->checkSolution(result_map, test_function);

  delete matrixGenerator;
  delete result_map;

  return 0;
}
