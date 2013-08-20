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
#include <stdio.h>
namespace D3{
class Tier {

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
	int tier_matrix_size;

	double** tier_matrix;
	double* tier_rhs;

public:
	Tier(Element* bot_left_near_element, Element* bot_left_far_element, Element* bot_right_far_element, Element* top_left_near_element,
			Element* top_left_far_element, Element* top_right_far_element, Element* top_right_near_element, Element* bot_right_near_element, ITripleArgFunction* f) :
	bot_left_far_element(bot_left_far_element),bot_left_near_element(bot_left_near_element), bot_right_far_element(bot_right_far_element), bot_right_near_element(bot_right_near_element),
	top_left_near_element(top_left_near_element), top_left_far_element(top_left_far_element), top_right_far_element(top_right_far_element), top_right_near_element(top_right_near_element)
	{

				tier_matrix_size = 117;
				tier_matrix = new double*[tier_matrix_size];
				for(int i = 0; i<tier_matrix_size; i++)
					tier_matrix[i] = new double[tier_matrix_size]();
				tier_rhs = new double[tier_matrix_size]();
				start_nr_adj = bot_left_near_element->get_bot_left_near_vertex_nr();
				bot_left_near_element->fillTierMatrix(tier_matrix,tier_rhs,f,start_nr_adj);
				bot_left_far_element->fillTierMatrix(tier_matrix,tier_rhs,f,start_nr_adj);
				bot_right_far_element->fillTierMatrix(tier_matrix,tier_rhs,f,start_nr_adj);
				if(bot_right_near_element != NULL)
					bot_right_near_element->fillTierMatrix(tier_matrix,tier_rhs,f,start_nr_adj);
				top_left_near_element->fillTierMatrix(tier_matrix,tier_rhs,f,start_nr_adj);
				top_left_far_element->fillTierMatrix(tier_matrix,tier_rhs,f,start_nr_adj);
				top_right_far_element->fillTierMatrix(tier_matrix,tier_rhs,f,start_nr_adj);
				top_right_near_element->fillTierMatrix(tier_matrix,tier_rhs,f,start_nr_adj);

	}

	void FillMatrixAndRhs(double** matrix, double* rhs, int matrix_size);

	virtual ~Tier()
	{
		delete[] tier_rhs;
		for(int i = 0; i<tier_matrix_size; i++){
			delete[] tier_matrix[i];
		}
		delete[] tier_matrix;
		delete bot_left_near_element;
		delete bot_left_far_element;
		delete bot_right_near_element;
		delete bot_right_far_element;
		delete top_left_near_element;
		delete top_left_far_element;
		delete top_right_near_element;
		delete top_right_far_element;

	}

	double** get_tier_matrix(){
		return tier_matrix;
	}

	double* get_tier_rhs(){
		return tier_rhs;
	}
};
}
#endif /* TIER_3D_H_ */
