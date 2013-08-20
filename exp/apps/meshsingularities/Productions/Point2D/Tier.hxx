/*
 * Tier.h
 *
 *  Created on: Aug 5, 2013
 *      Author: dgoik
 */

#ifndef TIER_2D_H_
#define TIER_2D_H_

#include <stdlib.h>
#include "Element.hxx"
#include "DoubleArgFunction.hxx"
#include "../EquationSystem.h"

namespace D2{
class Tier : EquationSystem
{

private:
	Element* bot_left_element;
	Element* top_left_element;
	Element* top_right_element;
	Element* bot_right_element;

	int start_nr_adj;
	static const int tier_matrix_size = 21;
	//int tier_matrix_size;

	//double** tier_matrix;
	//double* tier_rhs;

public:
	Tier(Element* top_left_element, Element* top_right_element, Element* bot_left_element, Element* bot_right_element, IDoubleArgFunction* f) : EquationSystem(tier_matrix_size),
	bot_left_element(bot_left_element), bot_right_element(bot_right_element), top_left_element(top_left_element), top_right_element(top_right_element)
	{

		start_nr_adj = bot_left_element->get_bot_left_vertex_nr();

		bot_left_element->fillTierMatrix(matrix, rhs, f, start_nr_adj);
		top_left_element->fillTierMatrix(matrix, rhs, f, start_nr_adj);
		if(bot_right_element != NULL)
			bot_right_element->fillTierMatrix(matrix, rhs, f, start_nr_adj);
		top_right_element->fillTierMatrix(matrix, rhs, f, start_nr_adj);

				/*
				tier_matrix_size = 21;
				tier_matrix = new double*[tier_matrix_size];
				for(int i = 0; i<tier_matrix_size; i++)
					tier_matrix[i] = new double[tier_matrix_size]();
				tier_rhs = new double[tier_matrix_size]();
				start_nr_adj = bot_left_element->get_bot_left_vertex_nr();

				bot_left_element->fillTierMatrix(tier_matrix, tier_rhs, f, start_nr_adj);
				top_left_element->fillTierMatrix(tier_matrix, tier_rhs, f, start_nr_adj);
				if(bot_right_element != NULL)
					bot_right_element->fillTierMatrix(tier_matrix, tier_rhs, f, start_nr_adj);
				top_right_element->fillTierMatrix(tier_matrix, tier_rhs, f, start_nr_adj);
				*/
	}

	void FillMatrixAndRhs(double** matrix, double* rhs, int matrix_size);

	virtual ~Tier()
	{
		delete bot_left_element;
		delete bot_right_element;
		delete top_left_element;
		delete top_right_element;

	}

	double** get_tier_matrix(){
		return matrix;
	}

	double* get_tier_rhs(){
		return rhs;
	}
};
}
#endif /* TIER_2D_H_ */
