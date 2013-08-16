/*
 * Tier.cpp
 *
 *  Created on: Aug 5, 2013
 *      Author: dgoik
 */

#include "Tier.hxx"
using namespace D2;
void Tier::FillMatrixAndRhs(double** matrix, double* rhs, int matrix_size)
{
		for(int i=0; i<tier_matrix_size; i++)
			for(int j=0; j<tier_matrix_size; j++){
				if(i+start_nr_adj < matrix_size && j+start_nr_adj < matrix_size)
				matrix[i+start_nr_adj][j+start_nr_adj] += tier_matrix[i][j];
			}

		for(int i =0; i<tier_matrix_size; i++)
			if(i+start_nr_adj < matrix_size)
				rhs[i+start_nr_adj] += tier_rhs[i];
}
