#include "EquationSystem.h"
#include <cmath>
EquationSystem::EquationSystem(int n)
{
	this->n = n;

	// we are working on continuous area of memory

	matrix = new double*[n];
	matrix[0] = new double[n*n]();
	for (int i = 0; i < n; ++i) {
		matrix[i] = matrix[0] + i * n;
	}

	rhs = new double[n]();
}

EquationSystem::EquationSystem(double ** matrix, double *rhs, int size)
{
	this->n = size;

	// we are working on continuous area of memory

	this->matrix = new double*[n];
	//this->matrix[0] = new double[n * n]();
	this->matrix[0] = new double[n*n]();
	for (int i = 1; i < n; ++i) {
		//this->matrix[i] = this->matrix[0] + i * n;
		this->matrix[i] = this->matrix[0] + i*n;
	}

	this->rhs = new double[n]();

	for (int i=0; i<size; ++i) {
		for (int j=0; j<size; ++j) {
			this->matrix[i][j] = matrix[i][j];
		}
		this->rhs[i] = rhs[i];
	}

}

EquationSystem::~EquationSystem()
{
	delete [] matrix[0];
	delete [] matrix;
	delete [] rhs;

	matrix = (double**) NULL;
	rhs = (double*) NULL;
}

void EquationSystem::eliminate(int rows)
{
	double x;
	double maxX;
	int maxRow;
	int i, j, k;

	for (i=0;i<rows;i++) {
		maxX = fabs(matrix[i][i]);
		maxRow = i;

		for (int k=i+1; k<rows; k++) {
			if (fabs(matrix[k][i]) > maxX) {
				maxX = fabs(matrix[k][i]);
				maxRow = k;
			}
		}

		if (maxRow != i) {
			swapRows(i, maxRow);
		}

		x = matrix[i][i];
		matrix[i][i] = 1.0;

		for (j=i+1;j<n;j++) {
			// on diagonal - only 1.0
			matrix[i][j] /= x;
		}

		rhs[i] /= x;

		for (j=i+1; j<n; j++) {
			x = matrix[j][i];
			for (k=i; k<n; k++) {
				matrix[j][k] -= x*matrix[i][k];
			}
			rhs[j] -= x*rhs[i];
			matrix[j][i] = 0.0;
		}
	}
}

void EquationSystem::backwardSubstitute(int startingRow)
{
	int i, j;
	double sum;

	for (i=startingRow; i>=0; --i) {
		sum = rhs[i];
		for (j=n-1;j>=i+1;--j) {
			sum -= matrix[i][j] * rhs[j];
			matrix[i][j] = 0.0;
		}
		rhs[i] = sum / matrix[i][i];
	}
}


void EquationSystem::swapCols(int i, int j)
{
	double tmp;
	int k;

	for (k=0; k<n; k++) {
		tmp = matrix[k][i];
		matrix[k][i] = matrix[k][j];
		matrix[k][j] = tmp;
	}
}

void EquationSystem::swapRows(int i, int j)
{
	double tmp;
	int k;

	for (k = 0; k<n; k++) {
		tmp = matrix[i][k];
		matrix[i][k] = matrix[j][k];
		matrix[j][k] = tmp;
	}

	tmp = rhs[i];
	rhs[i] = rhs[j];
	rhs[j] = tmp;
}

void EquationSystem::print()
{
	for (int i=0; i<n; i++) {
		for (int j=0; j<n; j++) {
			std::printf("% .15f ", matrix[i][j]);
		}
		std::printf (" | % .15f\n", rhs[i]);
	}
}

