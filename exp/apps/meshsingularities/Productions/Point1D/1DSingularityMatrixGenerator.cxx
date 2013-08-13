#include <stdlib.h>
#include <stdio.h>
#include "1DSingularityMatrixGenerator.hxx"

void get_matrix_and_rhs(int matrix_size, double*** matrix_p, double** rhs_p,  double l_boundary_condition, double r_boundary_condition){
	
	double** matrix = *matrix_p; 
	double* rhs = *rhs_p; 
	matrix = new double*[matrix_size];
	for(int i = 0; i<matrix_size; i++)
		matrix[i] = new double[matrix_size]();
	rhs = new double[matrix_size]();	
			
	for(int i = 1; i<matrix_size - 1; i++){
		matrix[i][i-1] = -1; 
		matrix[i][i] = 2; 
		matrix[i][i+1] = -1; 
		rhs[i] = 0;  

	}
	
	matrix[0][0] = 1; 
	matrix[matrix_size - 1][matrix_size - 1] = 1; 
	rhs[0] = l_boundary_condition; 
	rhs[matrix_size-1] = r_boundary_condition;
	
	//prepare rows (second and the row before last) - cholesky factorizatin requires symetry
	
	matrix[1][0] = 0; 
	rhs[1] = l_boundary_condition; 
	
	matrix[matrix_size - 2][matrix_size - 1] = 0; 
	rhs[matrix_size - 2] = r_boundary_condition;

	*matrix_p = matrix; 
	*rhs_p = rhs;  
}
