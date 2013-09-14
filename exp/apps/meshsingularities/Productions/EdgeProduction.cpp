/*
 * EdgeProduction.cpp
 *
 *  Created on: Sep 4, 2013
 *      Author: dgoik
 */

#include "EdgeProduction.h"
#include <stdio.h>


void EdgeProduction::Execute(EProduction productionToExecute, Vertex* v, EquationSystem* input)
{
	switch (productionToExecute) {
	case EProduction::B:
		B(v, input);
		break;
	case EProduction::C:
		C(v, input);
		break;
	case EProduction::D:
		D(v, input);
		break;
	case EProduction::MB:
		MB(v);
		break;
	case EProduction::MC:
		MC(v);
		break;
	case EProduction::MD:
		MD(v);
		break;
	case EProduction::MBLeaf:
		MBLeaf(v);
		break;
	case EProduction::MBC:
		MBC(v,false);
		break;
	case EProduction::MBRoot:
		MBC(v,true);
		break;
	default:
		printf("Invalid production!\n");
		break;
	}

}

void Pre(Vertex* v, EquationSystem* inData, int offset, bool eliminate)
{
	double** const sys_matrix = v->system->matrix;
	double* const sys_rhs = v->system->rhs;
	int size = v->system->n;
	const double** const in_matrix = (const double**) inData->matrix;
	const double* const in_rhs = (const double *) inData->rhs;

	int interfaceSize = size - offset;
	for(int i = 0; i<interfaceSize; i++){
		for(int j = 0; j<interfaceSize; j++)
			sys_matrix[i+offset][j+offset] = in_matrix[i][j];
		sys_rhs[i+offset] = in_rhs[i];
	}

	for(int i = interfaceSize; i<size; i++){
		for(int j = 0; j<interfaceSize; j++)
			sys_matrix[i - interfaceSize][j+offset] = in_matrix[i][j];
		sys_rhs[i - interfaceSize] = in_rhs[i];
	}

	for(int i = 0; i<interfaceSize; i++)
		for(int j = interfaceSize; j<size; j++)
			sys_matrix[i+offset][j-interfaceSize] = in_matrix[i][j];

	for(int i = interfaceSize; i<size; i++)
		for(int j = interfaceSize; j<size; j++)
			sys_matrix[i-interfaceSize][j-interfaceSize] = in_matrix[i][j];
	if(eliminate)
		v->system->eliminate(offset);

}

void EdgeProduction::B(Vertex* v, EquationSystem* inData) const
{
	Pre(v,inData,bOffset,true);
}

void EdgeProduction::C(Vertex* v, EquationSystem* inData) const
{
	Pre(v,inData,cOffset,true);
}

void EdgeProduction::D(Vertex* v, EquationSystem* inData) const
{
	double** const sys_matrix = v->system->matrix;
	double* const sys_rhs = v->system->rhs;

	const double** const in_matrix = (const double**) inData->matrix;
	const double* const in_rhs = (const double *) inData->rhs;

	Pre(v,inData,1,false);
	v->system->swapCols(4,1);
	v->system->swapRows(4,1);
	v->system->swapCols(4,2);
	v->system->swapRows(4,2);
	v->system->swapCols(4,3);
	v->system->swapRows(4,3);
	v->system->eliminate(2);

}

void EdgeProduction::Copy(Vertex* v, EquationSystem* inData) const
{
	for(int i = 0; i< inData->n; i++)
	{
		for(int j = 0; j<inData->n; j++)
			v->system->matrix[i][j] = inData->matrix[i][j];

		v->system->rhs[i] = inData->rhs[i];
	}


}

void EdgeProduction::MB(Vertex *v) const
{
	double** const matrix = v->system->matrix;
	double* const rhs = v->system->rhs;
	double system_size = v->system->n;

	const double** const left_matrix = (const double **) v->left->system->matrix;
	const double** const right_matrix = (const double **) v->right->system->matrix;
	const double* const left_rhs = (const double *) v->left->system->rhs;
	const double* const right_rhs = (const double *) v->right->system->rhs;


	int offset = 3;
	int common_b = 2;
	int common_b_growth = 2;
	int separate_b = 5;
	int separate_b_growth = 2;
	int common_bc = 1;
	int adaptation_lvl = system_size - common_b - 2*separate_b + common_bc;
	div_t lvl = div(adaptation_lvl, common_b_growth + 2*separate_b_growth);
	adaptation_lvl = lvl.quot;

	int nr_of_rows_to_eliminate  = common_b + common_b_growth*adaptation_lvl;

	int current_common_b = common_b + common_b_growth*adaptation_lvl;
	int current_separate_b = separate_b + separate_b_growth*adaptation_lvl;

	//1 1
	for(int i = 0; i<current_common_b; i++)
	{
		for(int j = 0; j<current_common_b; j++)
		{
			matrix[i][j] = right_matrix[i+offset+current_separate_b][j+offset+current_separate_b];
			matrix[i][j] += left_matrix[offset + current_common_b - 1 - i][offset + current_common_b - 1 - j];
		}

		rhs[i] = right_rhs[offset + i + current_separate_b];
		rhs[i] += left_rhs[offset + current_common_b - 1 - i];
	}
	//1 2 i 2 1
	for(int i = current_common_b; i<current_common_b + current_separate_b; i++)
	{
		for(int j = 0; j<current_common_b; j++)
		{
			matrix[i][j] = right_matrix[offset + i - current_common_b][offset + j + current_separate_b];
			matrix[j][i] = right_matrix[offset + j + current_separate_b][offset + i - current_common_b];
		}
		rhs[i] += right_rhs[offset + i - current_common_b];
	}

	//2 2
	for(int i = current_common_b; i<current_common_b + current_separate_b; i++)
		for(int j = current_common_b; j<current_common_b + current_separate_b; j++)
			matrix[i][j] = right_matrix[offset + i-current_common_b][offset + j-current_common_b];

	//1 3 3 1
	for(int i = 0 ; i<current_common_b; i++)
	{
		for(int j = current_common_b + current_separate_b - common_bc; j<system_size; j++)
		{
			matrix[i][j] += left_matrix[offset + current_common_b - i - 1][offset + j - current_separate_b + common_bc];
			matrix[j][i] += left_matrix[offset + j - current_separate_b + common_bc][offset + current_common_b - i - 1];
		}
	}

	//3 3
	for(int i = current_common_b + current_separate_b - common_bc; i<system_size; i++)
	{
		for(int j = current_common_b + current_separate_b - common_bc; j<system_size; j++)
		{
			matrix[i][j] += left_matrix[offset + i - current_separate_b + common_bc][offset + j - current_separate_b + common_bc];
		}
		rhs[i] += left_rhs[offset + i - current_separate_b + common_bc];
	}

	v->system->eliminate(current_common_b);

}

void EdgeProduction::BSMB(Vertex *v) const
{
	double* const rhs = v->system->rhs;
	double system_size = v->system->n;

	double* left_rhs = (double *) v->left->system->rhs;
	double* right_rhs = (double *) v->right->system->rhs;

	int offset = 3;
	int common_b = 2;
	int common_b_growth = 2;
	int separate_b = 5;
	int separate_b_growth = 2;
	int common_bc = 1;
	int adaptation_lvl = system_size - common_b - 2*separate_b + common_bc;
	div_t lvl = div(adaptation_lvl, common_b_growth + 2*separate_b_growth);
	adaptation_lvl = lvl.quot;

	int nr_of_rows_to_eliminate  = common_b + common_b_growth*adaptation_lvl;

	int current_common_b = common_b + common_b_growth*adaptation_lvl;
	int current_separate_b = separate_b + separate_b_growth*adaptation_lvl;

	for(int i = 0; i<current_common_b; i++)
	{
		right_rhs[offset + i + current_separate_b] = rhs[i];
		left_rhs[offset + current_common_b - 1 - i] = rhs[i];
	}

	for(int i = current_common_b; i<current_common_b + current_separate_b; i++)
		right_rhs[offset + i - current_common_b] = rhs[i];

	for(int i = current_common_b + current_separate_b - common_bc; i<system_size; i++)
		left_rhs[offset + i - current_separate_b + common_bc] = rhs[i];

}


void Mp1(Vertex* v, int offset, int non_separate_c_length, int adj)
{
	double** const matrix = v->system->matrix;
	double* const rhs = v->system->rhs;
	double system_size = v->system->n;

	const double** const left_matrix = (const double **) v->left->system->matrix;
	const double** const right_matrix = (const double **) v->right->system->matrix;
	const double* const left_rhs = (const double *) v->left->system->rhs;
	const double* const right_rhs = (const double *) v->right->system->rhs;

	// 1 1
	for(int i = 0; i<non_separate_c_length; i++)
	{
		for(int j = 0; j<non_separate_c_length; j++)
		{
			//printf("%d %d\n",offset + i + 5,offset + j + 5);
			matrix[i][j] = right_matrix[offset + i + 5 + adj][offset+ j + 5 + adj];
		}

		rhs[i] = right_rhs[offset + i + 5 + adj];
	}
	// 1 1
	for(int i = 0; i<2; i++)
	{
		for(int j = 0; j<2; j++)
			matrix[i][j] += left_matrix[offset + 1 - i][offset + 1 - j];
		rhs[i] += left_rhs[offset + 1 - i];
	}
	//1 2 2 1
	for(int i = 0; i<non_separate_c_length; i++)
	{
		for(int j = non_separate_c_length; j < non_separate_c_length + 5 + adj; j++)
		{
			matrix[i][j] = right_matrix[offset + i + 5 + adj][offset + j - non_separate_c_length];
			matrix[j][i] = right_matrix[offset + j - non_separate_c_length][offset + i + 5 + adj];
		}

	}
	//2 2
	for(int i = non_separate_c_length; i<non_separate_c_length + 5 + adj; i++)
	{
		for(int j = non_separate_c_length; j<non_separate_c_length + 5 + adj; j++)
			matrix[i][j] = right_matrix[offset + i - non_separate_c_length][offset + j - non_separate_c_length];
		rhs[i] = right_rhs[offset + i - non_separate_c_length];

	}

}

void Mp1Bs(Vertex* v, int offset, int non_separate_c_length, int adj)
{

	double* const rhs = v->system->rhs;

	double* const left_rhs = (double *) v->left->system->rhs;
	double* const right_rhs = (double *) v->right->system->rhs;

	for(int i = 0; i<non_separate_c_length; i++)
		right_rhs[offset + i + 5 + adj] = rhs[i];

	for(int i = 0; i<2; i++)
		left_rhs[offset + 1 - i] = rhs[i];

	for(int i = non_separate_c_length; i<non_separate_c_length + 5 + adj; i++)
		right_rhs[offset + i - non_separate_c_length] = rhs[i];
}

void Mp2(Vertex* v, int offset, int non_separate_c_length)
{

	double** const matrix = v->system->matrix;
	double* const rhs = v->system->rhs;
	double system_size = v->system->n;

	const double** const left_matrix = (const double **) v->left->system->matrix;
	const double* const left_rhs = (const double *) v->left->system->rhs;


	for(int i = 0; i<2; i++)
	{
			for(int j = 8 - non_separate_c_length; j<11; j++)
			{
				matrix[i][j] += left_matrix[offset + 1 - i][offset + j - 6 + non_separate_c_length];
				matrix[j][i] += left_matrix[offset + j - 6 + non_separate_c_length][offset + 1 - i];
			}
	}

	for(int i = 8 - non_separate_c_length; i<11; i++)
	{
		for(int j = 8 - non_separate_c_length; j<11; j++)
			matrix[i][j] += left_matrix[offset + i - 6 + non_separate_c_length][offset + j - 6 + non_separate_c_length];
		rhs[i] += left_rhs[offset + i - 6 + non_separate_c_length];
	}
}

void Mp2Bs(Vertex* v, int offset, int non_separate_c_length)
{
	double* const rhs = v->system->rhs;
	double* const left_rhs = v->left->system->rhs;

	for(int i = 8 - non_separate_c_length; i<11; i++)
	{
		left_rhs[offset + i - 6 + non_separate_c_length] = rhs[i];
	}
}

void EdgeProduction::MD(Vertex* v) const
{
	int offset = 2;
	int non_separate_c_length = 3;
	Mp1(v,offset,non_separate_c_length,-1);
	non_separate_c_length = 2;
	Mp2(v,offset,non_separate_c_length);

	int common_c = 1;
	v->system->eliminate(common_c);
}

void EdgeProduction::BSMD(Vertex* v) const
{
	int offset = 2;
	int non_separate_c_length = 3;
	Mp1Bs(v,offset,non_separate_c_length,-1);
	non_separate_c_length = 2;
	Mp2Bs(v,offset,non_separate_c_length);
}

void EdgeProduction::MC(Vertex* v) const
{
	int offset = cOffset;
	int non_separate_c_length = 3;

	Mp1(v,offset,non_separate_c_length,0);
	Mp2(v,offset,non_separate_c_length);

	int common_c = 1;
	v->system->eliminate(common_c);

}

void EdgeProduction::BSMC(Vertex* v) const
{
	int offset = cOffset;
	int non_separate_c_length = 3;

	Mp1Bs(v,offset,non_separate_c_length,0);
	Mp2Bs(v,offset,non_separate_c_length);
}


void EdgeProduction::MBLeaf(Vertex* v) const
{
	int offset = 2;
	int non_separate_c_length = 2;

	Mp1(v,offset,non_separate_c_length,0);
	Mp2(v,offset,non_separate_c_length);

	int common_c = 2;
	v->system->eliminate(common_c);
}

void EdgeProduction::BSMBLeaf(Vertex* v) const
{
	int offset = 2;
	int non_separate_c_length = 2;

	Mp1Bs(v,offset,non_separate_c_length,0);
	Mp2Bs(v,offset,non_separate_c_length);
}

void EdgeProduction::MBC(Vertex* v, bool root) const
{
	double** const matrix = v->system->matrix;
	double* const rhs = v->system->rhs;
	double system_size = v->system->n;

	const double** const left_matrix = (const double **) v->left->system->matrix;
	const double** const right_matrix = (const double **) v->right->system->matrix;
	double* const left_rhs = v->left->system->rhs;
	double* const right_rhs = v->right->system->rhs;

	int first_mbc_system_size = 14;
	int adaptation_lvl = system_size - first_mbc_system_size;
	div_t lvl = div(adaptation_lvl, 4);
	adaptation_lvl = lvl.quot;

	//mc md is always right mb mbleaf is always left
	int mc_offset = 1;
	int mb_offset = 2 + 2*adaptation_lvl;
	int mb_growth = adaptation_lvl*2;


	//9 x 9 square, always only first 3 rows to eliminate
	int mc_elimination_row_nrs[3];
	mc_elimination_row_nrs[0] = mc_offset + 1;
	mc_elimination_row_nrs[1] = mc_offset;
	mc_elimination_row_nrs[2] = mc_offset + 9;

	for(int i = 0; i<3; i++)
	{
		for(int j = 0; j<3; j++)
		{
			matrix[i][j] = left_matrix[mb_offset + i + 3 + mb_growth][mb_offset + j + 3 + mb_growth];
			matrix[i][j] += right_matrix[mc_elimination_row_nrs[i]][mc_elimination_row_nrs[j]];
		}
		rhs[i] = left_rhs[mb_offset + i + 3 + mb_growth] + right_rhs[mc_elimination_row_nrs[i]];
	}

	//rest of first 3 rows
	for(int i = 0; i<3; i++)
	{
		for(int j = 3; j<6 + mb_growth; j++)
		{
			matrix[i][j] = left_matrix[mb_offset + i + 3 + mb_growth][mb_offset + j - 3];
			matrix[j][i] = left_matrix[mb_offset + j - 3][mb_offset + i + 3 + mb_growth];
		}

		for(int j = 5 + mb_growth; j<12 + mb_growth; j++)
		{
			matrix[i][j] += right_matrix[mc_elimination_row_nrs[i]][mc_offset + 2 + j - 5 - mb_growth];
			matrix[j][i] += right_matrix[mc_offset + 2 + j - 5 - mb_growth][mc_elimination_row_nrs[i]];
		}

		for(int j = 11 + mb_growth; j < 11 + mb_growth + 3 + mb_growth; j++)
		{
			matrix[i][j] += left_matrix[mb_offset + i + 3 + mb_growth][mb_offset + 6 + j - 11];
			matrix[j][i] += left_matrix[mb_offset + 6 + j - 11][mb_offset + i + 3 + mb_growth];
		}

	}

	for(int i = 3; i < 3 + 3 + mb_growth; i++)
	{
		for(int j = 3; j < 3 + 3 + mb_growth; j++)
			matrix[i][j] = left_matrix[mb_offset - 3 + i][mb_offset - 3 + j];

		for(int j = 11 + mb_growth; j < 11 + mb_growth + 3 + mb_growth; j++)
		{
			matrix[i][j] = left_matrix[mb_offset - 3 + i][mb_offset + 6 + j - 11];
			matrix[j][i] = left_matrix[mb_offset + 6 + j - 11][mb_offset - 3 + i];
		}

		rhs[i] = left_rhs[mb_offset - 3 + i];
	}

	for(int i = 5 + mb_growth; i < 12 + mb_growth; i++)
	{
		for(int j = 5 + mb_growth; j < 12 + mb_growth; j++)
		{
			matrix[i][j] += right_matrix[mc_offset + 2 + i - 5 - mb_growth][mc_offset + 2 + j - 5 - mb_growth];
		}
		rhs[i] += right_rhs[mc_offset + 2 + i - 5 - mb_growth];
	}

	for(int i = 11 + mb_growth; i < 11 + mb_growth + 3 + mb_growth; i++)
	{
		for(int j = 11 + mb_growth; j < 11 + mb_growth + 3 + mb_growth; j++)
		{
			matrix[i][j] += left_matrix[mb_offset + 6 + i - 11][mb_offset + 6 + j - 11];
		}
		rhs[i] += left_rhs[mb_offset + 6 + i - 11];
	}

	if(root){
		v->system->eliminate(v->system->n);
		v->system->backwardSubstitute(v->system->n-1);
		v->system->print();
		//backward substitution
		for(int i = 0; i<3; i++)
		{
			left_rhs[mb_offset + i + 3 + mb_growth] = rhs[i];
			right_rhs[mc_elimination_row_nrs[i]] = rhs[i];
		}

		for(int i = 3; i < 3 + 3 + mb_growth; i++)
			left_rhs[mb_offset - 3 + i] = rhs[i];

		for(int i = 5 + mb_growth; i < 12 + mb_growth; i++)
			right_rhs[mc_offset + 2 + i - 5 - mb_growth] = rhs[i];

		for(int i = 11 + mb_growth; i < 11 + mb_growth + 3 + mb_growth; i++)
			 left_rhs[mb_offset + 6 + i - 11] = rhs[i];

	}
	else
	{
		v->system->eliminate(3);
	}
}

void EdgeProduction::BSMBC(Vertex* v) const
{
	double* const rhs = v->system->rhs;
	double system_size = v->system->n;

	double* const left_rhs = v->left->system->rhs;
	double* const right_rhs =  v->right->system->rhs;

	int first_mbc_system_size = 14;
	int adaptation_lvl = system_size - first_mbc_system_size;
	div_t lvl = div(adaptation_lvl, 4);
	adaptation_lvl = lvl.quot;

	//mc md is always right mb mbleaf is always left
	int mc_offset = 1;
	int mb_offset = 2 + 2*adaptation_lvl;
	int mb_growth = adaptation_lvl*2;

	int mc_elimination_row_nrs[3];
	mc_elimination_row_nrs[0] = mc_offset + 1;
	mc_elimination_row_nrs[1] = mc_offset;
	mc_elimination_row_nrs[2] = mc_offset + 9;

	//backward substitution
	for(int i = 0; i<3; i++)
	{
		left_rhs[mb_offset + i + 3 + mb_growth] = rhs[i];
		right_rhs[mc_elimination_row_nrs[i]] = rhs[i];
	}

	for(int i = 3; i < 3 + 3 + mb_growth; i++)
		left_rhs[mb_offset - 3 + i] = rhs[i];

	for(int i = 5 + mb_growth; i < 12 + mb_growth; i++)
		right_rhs[mc_offset + 2 + i - 5 - mb_growth] = rhs[i];

	for(int i = 11 + mb_growth; i < 11 + mb_growth + 3 + mb_growth; i++)
		 left_rhs[mb_offset + 6 + i - 11] = rhs[i];

}
