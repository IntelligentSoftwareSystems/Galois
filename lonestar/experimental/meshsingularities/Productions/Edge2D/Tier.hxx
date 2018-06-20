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
 * Tier.hxx
 *
 *  Created on: Aug 27, 2013
 *      Author: dgoik
 */

#ifndef TIER_2D_EDGE_HXX_
#define TIER_2D_EDGE_HXX_

#include <stdlib.h>
#include "../Point2D/Element.hxx"
#include "../Point2D/DoubleArgFunction.hxx"
#include "../EquationSystem.h"
#include <vector>

using namespace D2;
namespace D2Edge {

class Tier : public EquationSystem {

private:
  Element* element;
  IDoubleArgFunction* f;
  double** global_matrix;
  double* global_rhs;
  static const int tier_matrix_size = 9;

public:
  Tier(Element* element, IDoubleArgFunction* f, double** global_matrix,
       double* global_rhs)
      : EquationSystem(tier_matrix_size), element(element), f(f),
        global_matrix(global_matrix), global_rhs(global_rhs) {}

  void InitTier() {

    // element->fillMatrices(NULL,global_matrix,NULL,global_rhs,f,0);

    int global_numbers[9];
    element->get_nrs(global_numbers);

    int local_numbers[9];
    for (int i = 0; i < 9; i++)
      local_numbers[i] = i;

    element->set_nrs(local_numbers);

    element->fillMatrices(matrix, NULL, rhs, NULL, f, 0);
    element->set_nrs(global_numbers);
  }

  void FillNumberPairs(std::map<std::pair<int, int>, double>* map,
                       double* _rhs) {
    int global_numbers[9];
    element->get_nrs(global_numbers);

    std::map<std::pair<int, int>, double>::iterator it;
    for (int i = 0; i < 9; i++) {
      for (int j = i; j < 9; j++) {
        int i_indx = global_numbers[i];
        int j_indx = global_numbers[j];
        if (global_numbers[i] > global_numbers[j]) {
          i_indx = global_numbers[j];
          j_indx = global_numbers[i];
        }
        std::pair<int, int> key(i_indx, j_indx);
        it = map->find(key);
        if (it == map->end())
          (*map)[key] = matrix[i][j];
        else
          it->second += matrix[i][j];
      }

      _rhs[global_numbers[i]] += rhs[i];
    }
  }

  void add_results(std::vector<double>* result) {
    int global_numbers[9];
    element->get_nrs(global_numbers);
    for (int i = 0; i < 9; i++) {
      (*result)[global_numbers[i]] = rhs[i];
    }
  }

  virtual ~Tier() { delete element; }

  double** get_tier_matrix() { return matrix; }

  double* get_tier_rhs() { return rhs; }
};
} // namespace D2Edge
  /* namespace D2Edge */
#endif /* TIER_2D_EDGE_HXX_ */
