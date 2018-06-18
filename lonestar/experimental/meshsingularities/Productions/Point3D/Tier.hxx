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
