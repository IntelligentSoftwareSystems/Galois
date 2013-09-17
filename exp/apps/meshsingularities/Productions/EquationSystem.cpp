#include "EquationSystem.h"
#include <cmath>

#include <Galois/Galois.h>
#include <boost/iterator.hpp>

EquationSystem::EquationSystem(unsigned long n)
{
	this->n = n;
	unsigned long i;

	// we are working on continuous area of memory

	matrix = new double*[n];
	matrix[0] = new double[n*(n+1)]();
	for (i = 0; i < n; ++i) {
		matrix[i] = matrix[0] + i * n;
	}

	if (matrix == NULL || matrix[0] == NULL) {
		throw std::string("Cannot allocate memory!");
	}

	rhs = matrix[0]+n*n;

	origPtr = matrix[0];
}

EquationSystem::EquationSystem(double ** matrix, double *rhs, unsigned long size)
{
	this->n = size;
	unsigned long i;

	// we are working on continuous area of memory

	this->matrix = new double*[n];
	this->matrix[0] = new double[n*(n+1)]();

	for (i = 1; i < n; ++i) {
		this->matrix[i] = this->matrix[0] + i*n;
	}

	if (matrix == NULL || matrix[0] == NULL) {
		throw std::string("Cannot allocate memory!");
	}


	this->rhs = this->matrix[0]+n*n;

	for (i=0; i<size; ++i) {
		for (int j=0; j<size; ++j) {
			this->matrix[i][j] = matrix[i][j];
		}
		this->rhs[i] = rhs[i];
	}

	origPtr = this->matrix[0];

}

EquationSystem::~EquationSystem()
{
	if (matrix != NULL) {
		delete [] origPtr;
		delete [] matrix;
	}

}

void EquationSystem::eliminate(const int rows)
{


	double maxX;
	register int maxRow;
	double x;
	int i, j;

	for (i=0;i<rows;++i) {
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

		x = matrix[i][i];
		// on diagonal - only 1.0
		matrix[i][i] = 1.0;

		for (j=i+1;j<n;++j) {
			matrix[i][j] /= x;
		}

		rhs[i] /= x;
		/*Galois::do_all(boost::counting_iterator<int>(i+1),
				boost::counting_iterator<int>(n), [&] (int j) {
				x = matrix[j][i];
				for (int k = i+1; k<n; ++k) {
					matrix[j][k] -= x*matrix[i][k];
				}
				rhs[j] -= x*rhs[i];
			}); */
		for (j=i+1; j<n; ++j) {
			x = matrix[j][i];

			for (int k = i+1; k<n; ++k) {
				matrix[j][k] -= x*matrix[i][k];
			}
			rhs[j] -= x*rhs[i];
		}
	}
}

void EquationSystem::backwardSubstitute(const int startingRow)
{
	for (int i=startingRow; i>=0; --i) {
		double sum = rhs[i];
		for (int j=n-1; j>=i+1;--j) {
			sum -= matrix[i][j] * rhs[j];
			matrix[i][j] = 0.0;
		}
		rhs[i] = sum;// / matrix[i][i]; // after elimination we have always 1.0 at matrix[i][i]
		// do not need to divide by matrix[i][i]
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

void EquationSystem::checkRow(int row_nr, int* values, int values_cnt)
{
	double v = 0;
	for(int i = 0; i<values_cnt; i++)
		v += matrix[row_nr][values[i]];

	printf("DIFF : %lf\n",rhs[row_nr] - v);
}

void EquationSystem::print() const
{
	for (int i=0; i<n; ++i) {
		for (int j=0; j<n; ++j) {
			std::printf("% .6f ", matrix[i][j]);
		}
		std::printf (" | % .6f\n", rhs[i]);
	}
}

