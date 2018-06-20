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
 * MatrixGenerator.hxx
 *
 *  Created on: Aug 22, 2013
 *      Author: dgoik
 */

#ifndef MATRIXGENERATOR_HXX_
#define MATRIXGENERATOR_HXX_
#include <cstdarg>
#include <boost/function.hpp>
#include <boost/bind.hpp>
#include <vector>
#include <map>
#include <stdio.h>
#include "../EquationSystem.h"
#include "../GaloisWorker.h"

class GenericMatrixGenerator {

protected:
  double** matrix;
  double* rhs;
  int matrix_size;
  std::vector<EquationSystem*>* tier_vector;

public:
  virtual std::vector<EquationSystem*>*
  CreateMatrixAndRhs(TaskDescription& task_description) = 0;
  virtual void checkSolution(std::map<int, double>* solution_map,
                             double (*f)(int dim, ...)) = 0;

  double** GetMatrix() { return matrix; }

  double* GetRhs() { return rhs; }

  int GetMatrixSize() { return matrix_size; }

  virtual std::vector<int>* GetProductionParameters(int polynomial_degree) {
    return new std::vector<int>();
  }

  virtual bool GetMumpsArrays(int*& in, int*& jn, double*& a, double*& rhs,
                              int& n, int& nz) {
    return false;
  }

  virtual ~GenericMatrixGenerator() {
    for (int i = 0; i < matrix_size; i++)
      delete[] matrix[i];
    delete[] matrix;
    delete[] rhs;

    std::vector<EquationSystem*>::iterator it_t = tier_vector->begin();
    for (; it_t != tier_vector->end(); ++it_t)
      delete *it_t;

    delete tier_vector;
  }
};

#endif /* MATRIXGENERATOR_HXX_ */
