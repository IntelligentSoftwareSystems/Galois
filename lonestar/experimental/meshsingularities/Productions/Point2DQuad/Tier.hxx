/*
 * Tier.hxx
 *
 *  Created on: Aug 27, 2013
 *      Author: dgoik
 */

#ifndef TIER_2D_QUAD_HXX_
#define TIER_2D_QUAD_HXX_

#include <stdlib.h>
#include "../Point2D/Element.hxx"
#include "../Point2D/DoubleArgFunction.hxx"
#include "../EquationSystem.h"
#include <stdio.h>
using namespace D2;
namespace D2Quad {

class Tier : public EquationSystem
{

private:
	Element** elements;

	int start_nr_adj;
	int tier_size;
	static const int tier_matrix_size = 72;

public:
	Tier(Element** elements, IDoubleArgFunction* f, double** global_matrix, double* global_rhs, int tier_size) :
		EquationSystem(tier_matrix_size), elements(elements), tier_size(tier_size)

	{
		start_nr_adj = elements[0]->get_bot_left_vertex_nr();
		for(int i = 0; i<tier_size; i++)
		{
			elements[i]->fillMatrices(matrix, global_matrix, rhs, global_rhs, f, start_nr_adj);
		}

	}

	virtual ~Tier()
	{
		delete[] elements;
	}

	double** get_tier_matrix(){
		return matrix;
	}

	double* get_tier_rhs(){
		return rhs;
	}
};
}
 /* namespace D2Quad */
#endif /* TIER_2D_QUAD_HXX_ */
