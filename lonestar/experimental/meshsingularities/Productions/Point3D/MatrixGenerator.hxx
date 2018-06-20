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

#ifndef __MATRIXGENERATOR_3D_H_INCLUDED
#define __MATRIXGENERATOR_3D_H_INCLUDED

#include "TripleArgFunction.hxx"
#include "Element.hxx"
#include "EPosition.hxx"
#include "Tier.hxx"
#include "../MatrixGeneration/GenericMatrixGenerator.hxx"
#include <map>
#include <vector>
namespace D3 {
class MatrixGenerator : public GenericMatrixGenerator {

private:
  std::list<Element*> element_list;

public:
  virtual std::vector<EquationSystem*>*
  CreateMatrixAndRhs(TaskDescription& task_description);
  virtual void checkSolution(std::map<int, double>* solution_map,
                             double (*f)(int dim, ...));

  virtual std::vector<int>* GetProductionParameters(int polynomial_degree) {
    std::vector<int>* vec = new std::vector<int>(4);
    (*vec)[0]             = 19;
    (*vec)[1]             = 75;
    (*vec)[2]             = 117;
    (*vec)[3]             = 83;
    return vec;
  }

  virtual ~MatrixGenerator() {
    std::list<Element*>::iterator it_e = element_list.begin();
    for (; it_e != element_list.end(); ++it_e) {
      delete *it_e;
    }
  }
};
} // namespace D3
#endif
