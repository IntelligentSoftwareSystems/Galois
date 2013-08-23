#include "EquationSystem.h"
#include <cmath>
EquationSystem::EquationSystem(int n)
{
	this->n = n;

	// we are working on continuous area of memory

	matrix = new double*[n];
	matrix[0] = new double[n*(n+1)]();
	for (int i = 0; i < n; ++i) {
		matrix[i] = matrix[0] + i * n;
	}

	rhs = matrix[0]+n*n;
}

EquationSystem::EquationSystem(double ** matrix, double *rhs, int size)
{
	this->n = size;

	// we are working on continuous area of memory

	this->matrix = new double*[n];
	this->matrix[0] = new double[n*(n+1)]();

	for (int i = 1; i < n; ++i) {
		this->matrix[i] = this->matrix[0] + i*n;
	}

	this->rhs = this->matrix[0]+n*n;

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

	matrix = (double**) 0;
	rhs = (double*) 0;
}

void EquationSystem::eliminate(const int rows)
{
	double maxX;
	int maxRow;
	double x;
	for (int i=0;i<rows;++i) {
		maxX = fabs(matrix[i][i]);
		maxRow = i;

		for (int k=i+1; k<rows; ++k) {
			if (fabs(matrix[k][i]) > maxX) {
				maxX = fabs(matrix[k][i]);
				maxRow = k;
			}
		}

		if (maxRow != i) {
			swapRows(i, maxRow);
		}

		double x = matrix[i][i];
		matrix[i][i] = 1.0;

		for (int j=i+1;j<n;++j) {
			// on diagonal - only 1.0
			matrix[i][j] /= x;
		}

		rhs[i] /= x;

		for (int j=i+1; j<n; ++j) {
			x = matrix[j][i];
			for (int k=i; k<n; ++k) {
				matrix[j][k] -= x*matrix[i][k];
			}
			rhs[j] -= x*rhs[i];
			//matrix[j][i] = 0.0;
		}
	}
}

void EquationSystem::backwardSubstitute(const int startingRow)
{
	for (int i=startingRow; i>=0; --i) {
		double sum = rhs[i];
		for (int j=n-1;j>=i+1;--j) {
			sum -= matrix[i][j] * rhs[j];
			matrix[i][j] = 0.0;
		}
		rhs[i] = sum / matrix[i][i];
	}
}

void EquationSystem::swapCols(const int i, const int j)
{
	for (int k=0; k<n; ++k) {
		double tmp = matrix[k][i];
		matrix[k][i] = matrix[k][j];
		matrix[k][j] = tmp;
	}
}

void EquationSystem::swapRows(const int i, const int j)
{
	// reduced complexity from O(n) to O(1)
	double tmp;
	double *tmpPtr = matrix[i];
	matrix[i] = matrix[j];
	matrix[j] = tmpPtr;

	tmp = rhs[i];
	rhs[i] = rhs[j];
	rhs[j] = tmp;
}

void EquationSystem::print() const
{
	for (int i=0; i<n; ++i) {
		for (int j=0; j<n; ++j) {
			std::printf("% .16f ", matrix[i][j]);
		}
		std::printf (" | % .16f\n", rhs[i]);
	}
}

