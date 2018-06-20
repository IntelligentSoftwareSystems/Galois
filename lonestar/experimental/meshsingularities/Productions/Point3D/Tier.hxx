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
 * Tier.h
 *
 *  Created on: Aug 5, 2013
 *      Author: dgoik
 */

#ifndef TIER_3D_H_
#define TIER_3D_H_

#include <stdlib.h>
#include "Element.hxx"
#include "TripleArgFunction.hxx"
#include "../EquationSystem.h"

namespace D3 {
class Tier : public EquationSystem {

private:
  Element* bot_left_near_element;
  Element* bot_left_far_element;
  Element* bot_right_far_element;
  Element* bot_right_near_element;
  Element* top_left_near_element;
  Element* top_left_far_element;
  Element* top_right_far_element;
  Element* top_right_near_element;

  int start_nr_adj;
  static const int tier_matrix_size = 117;

public:
  Tier(Element* bot_left_near_element, Element* bot_left_far_element,
       Element* bot_right_far_element, Element* top_left_near_element,
       Element* top_left_far_element, Element* top_right_far_element,
       Element* top_right_near_element, Element* bot_right_near_element,
       ITripleArgFunction* f, double** global_matrix, double* global_rhs)
      : EquationSystem(tier_matrix_size),
        bot_left_far_element(bot_left_far_element),
        bot_left_near_element(bot_left_near_element),
        bot_right_far_element(bot_right_far_element),
        bot_right_near_element(bot_right_near_element),
        top_left_near_element(top_left_near_element),
        top_left_far_element(top_left_far_element),
        top_right_far_element(top_right_far_element),
        top_right_near_element(top_right_near_element) {

    start_nr_adj = bot_left_near_element->get_bot_left_near_vertex_nr();
    bot_left_near_element->fillMatrices(matrix, global_matrix, rhs, global_rhs,
                                        f, start_nr_adj);
    bot_left_far_element->fillMatrices(matrix, global_matrix, rhs, global_rhs,
                                       f, start_nr_adj);
    bot_right_far_element->fillMatrices(matrix, global_matrix, rhs, global_rhs,
                                        f, start_nr_adj);
    if (bot_right_near_element != NULL)
      bot_right_near_element->fillMatrices(matrix, global_matrix, rhs,
                                           global_rhs, f, start_nr_adj);
    top_left_near_element->fillMatrices(matrix, global_matrix, rhs, global_rhs,
                                        f, start_nr_adj);
    top_left_far_element->fillMatrices(matrix, global_matrix, rhs, global_rhs,
                                       f, start_nr_adj);
    top_right_far_element->fillMatrices(matrix, global_matrix, rhs, global_rhs,
                                        f, start_nr_adj);
    top_right_near_element->fillMatrices(matrix, global_matrix, rhs, global_rhs,
                                         f, start_nr_adj);
  }

  // void FillMatrixAndRhs(double** matrix, double* rhs, int matrix_size);

  virtual ~Tier() {}

  double** get_tier_matrix() { return matrix; }

  double* get_tier_rhs() { return rhs; }
};
} // namespace D3
#endif /* TIER_3D_H_ */
