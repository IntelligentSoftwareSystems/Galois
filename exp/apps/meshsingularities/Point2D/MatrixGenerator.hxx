#ifndef __MATRIXGENERATOR_H_INCLUDED
#define __MATRIXGENERATOR_H_INCLUDED

#include "DoubleArgFunction.hxx"
#include "Element.hxx"
#include "EPosition.hxx"
#include <stack>

class MatrixGenerator {
		
	private:
		double** matrix; 
		double* rhs; 
		int matrix_size;
	public:
		void CreateMatrixAndRhs(int nr_of_tiers, double bot_left_x, double bot_left_y, double size, IDoubleArgFunction* f);
		
		double** GetMatrix()
		{
			return matrix;
		}

		double* GetRhs()
		{
			return rhs;
		}
		
		int GetMatrixSize()
		{
			return matrix_size;
		}	

};
#endif
