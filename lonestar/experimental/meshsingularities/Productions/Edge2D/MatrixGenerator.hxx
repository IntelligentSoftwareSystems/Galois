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

#ifndef __MATRIXGENERATOR_2DEDGE_H_INCLUDED
#define __MATRIXGENERATOR_2DEDGE_H_INCLUDED

#include "../Point2D/DoubleArgFunction.hxx"
#include "../Point2D/Element.hxx"
#include "../Point2D/EPosition.hxx"
#include "Tier.hxx"
#include "../MatrixGeneration/GenericMatrixGenerator.hxx"
#include <vector>
#include <map>
#include <cmath>
namespace D2Edge {
class MatrixGenerator : public GenericMatrixGenerator {

private:
  int elements_size;
  Element** elements;
  int nr_of_elements;
  void CreateTiers(int to_create, int element_id, double size,
                   double* coordinates, IDoubleArgFunction* f, bool first_tier);

  // mumps
  bool mumps_arrays_created;
  int* in;
  int* jn;
  double* a;
  int n;
  int nz;

public:
  virtual std::vector<EquationSystem*>*
  CreateMatrixAndRhs(TaskDescription& task_description);
  virtual void checkSolution(std::map<int, double>* solution_map,
                             double (*f)(int dim, ...));
  virtual bool GetMumpsArrays(int*& in, int*& jn, double*& a, double*& rhs,
                              int& n, int& nz);
  virtual std::vector<int>* GetProductionParameters(int polynomial_degree);

  virtual ~MatrixGenerator() {
    for (int i = 0; i < elements_size; i++)
      delete elements[i];
    delete[] elements;
  }
};
} // namespace D2Edge
#endif
