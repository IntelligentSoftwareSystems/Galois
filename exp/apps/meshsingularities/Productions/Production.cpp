#include "Production.h"

void AbstractProduction::A1(Vertex *v, EquationSystem *inData) const
{
	// here we assumed, that inData is currently pre-processed
	double** const sys_matrix = v->system->matrix;
	double* const sys_rhs = v->system->rhs;

	const double** const in_matrix = (const double**) inData->matrix;
	const double* const in_rhs = (const double *) inData->rhs;

	if (offset > 0) {
		for (int i=0; i<interfaceSize; ++i) {
			for (int j=0; j<interfaceSize; ++j) {
				sys_matrix[i+offset][j+offset] = in_matrix[i+a1Offset][j+a1Offset]; // 1
				sys_matrix[i+offset][j+offset+interfaceSize] = in_matrix[i+a1Offset][j+offset+interfaceSize+a1Offset]; // 3
				sys_matrix[i+offset+interfaceSize][j+offset] = in_matrix[i+offset+interfaceSize+a1Offset][j+a1Offset]; // 7
				sys_matrix[i+offset+interfaceSize][j+offset+interfaceSize] = in_matrix[i+offset+interfaceSize+a1Offset][j+offset+interfaceSize+a1Offset]; // 9
			}
			sys_rhs[i+offset] = in_rhs[i+a1Offset];
			sys_rhs[i+offset+interfaceSize] = in_rhs[i+offset+interfaceSize+a1Offset];
		}

		for (int i=0; i<offset; ++i) {
			for (int j=0; j<offset; ++j) {
				sys_matrix[i][j] = in_matrix[i+interfaceSize+a1Offset][j+interfaceSize+a1Offset]; // 5
			}
			sys_rhs[i] = in_rhs[i+interfaceSize+a1Offset];
		}

		for (int i=0; i<offset; ++i) {
			for (int j=0; j<interfaceSize; ++j) {
				sys_matrix[i][j+offset] = in_matrix[i+interfaceSize+a1Offset][j+a1Offset]; // 4
				sys_matrix[i][j+offset+interfaceSize] = in_matrix[i+interfaceSize+a1Offset][j+offset+interfaceSize+a1Offset]; // 6
				sys_matrix[j+offset][i] = in_matrix[j+a1Offset][i+interfaceSize+a1Offset]; // 2
				sys_matrix[j+offset+interfaceSize][i] = in_matrix[j+offset+interfaceSize+a1Offset][i+interfaceSize+a1Offset]; // 8
			}
		}

		v->system->eliminate(leafSize-2*interfaceSize);
	}
	else {
		// just copy data?

		for (int i=0; i<leafSize; ++i) {
			for (int j=0; j<leafSize; ++j) {
				sys_matrix[i][j] = in_matrix[i][j];
			}
			sys_rhs[i] = in_rhs[i];
		}

	}
}

void AbstractProduction::A(Vertex *v, EquationSystem *inData) const
{
	// You will probably need to overwrite this method
	// to adjust it to your requirements.

	double** const sys_matrix = v->system->matrix;
	double* const sys_rhs = v->system->rhs;

	const double** const in_matrix = (const double **) inData->matrix;
	const double* const in_rhs = (const double *) inData->rhs;

	if (offset > 0) {
		for (int i=0; i<interfaceSize; ++i) {
			for (int j=0; j<interfaceSize; ++j) {
				sys_matrix[i+offset][j+offset] = in_matrix[i][j]; // 1
				sys_matrix[i+offset][j+offset+interfaceSize] = in_matrix[i][j+offset+interfaceSize]; // 3
				sys_matrix[i+offset+interfaceSize][j+offset] = in_matrix[i+offset+interfaceSize][j]; // 7
				sys_matrix[i+offset+interfaceSize][j+offset+interfaceSize] = in_matrix[i+offset+interfaceSize][j+offset+interfaceSize]; // 9
			}
			sys_rhs[i+offset] = inData->rhs[i];
			sys_rhs[i+offset+interfaceSize] = in_rhs[i+offset+interfaceSize];
		}

		for (int i=0; i<offset; ++i) {
			for (int j=0; j<offset; ++j) {
				sys_matrix[i][j] = in_matrix[i+interfaceSize][j+interfaceSize]; // 5
			}
		}

		for (int i=0; i<offset; ++i) {
			for (int j=0; j<interfaceSize; ++j) {
				sys_matrix[i][j+offset] = in_matrix[i+interfaceSize][j]; // 4
				sys_matrix[i][j+offset+interfaceSize] = in_matrix[i+interfaceSize][j+offset+interfaceSize]; // 6
				sys_matrix[j+offset][i] = in_matrix[j][i+interfaceSize]; // 2
				sys_matrix[j+offset+interfaceSize][i] = in_matrix[j+offset+interfaceSize][i+interfaceSize]; // 8
			}
			sys_rhs[i] = in_rhs[i+interfaceSize];
		}

		v->system->eliminate(offset);

	}
	else {
		for (int i=0; i<interfaceSize*2; ++i) {
			for (int j=0; j<interfaceSize*2; ++j) {
				sys_matrix[i][j] = in_matrix[i][j];
			}
			sys_rhs[i] = in_rhs[i];
		}
	}

}

void AbstractProduction::AN(Vertex *v, EquationSystem *inData) const
{
	double** const sys_matrix = v->system->matrix;
	double* const sys_rhs = v->system->rhs;

	const double** const in_matrix = (const double **) inData->matrix;
	const double* const in_rhs = (const double *) inData->rhs;

	if (offset > 0) {
		for (int i=0; i<interfaceSize; ++i) {
			for (int j=0; j<interfaceSize; ++j) {
				sys_matrix[i+offset][j+offset] = in_matrix[i+anOffset][j+anOffset]; // 1
				sys_matrix[i+offset][j+offset+interfaceSize] = in_matrix[i+anOffset][j+offset+interfaceSize+anOffset]; // 3
				sys_matrix[i+offset+interfaceSize][j+offset] = in_matrix[i+offset+interfaceSize+anOffset][j+anOffset]; // 7
				sys_matrix[i+offset+interfaceSize][j+offset+interfaceSize] = in_matrix[i+offset+interfaceSize+anOffset][j+offset+interfaceSize+anOffset]; // 9
			}
			sys_rhs[i+offset] = in_rhs[i+anOffset];
			sys_rhs[i+offset+interfaceSize] = in_rhs[i+offset+interfaceSize+anOffset];
		}

		for (int i=0; i<offset; ++i) {
			for (int j=0; j<offset; ++j) {
				sys_matrix[i][j] = in_matrix[i+interfaceSize+anOffset][j+interfaceSize+anOffset]; // 5
			}
		}

		for (int i=0; i<offset; ++i) {
			for (int j=0; j<interfaceSize; ++j) {
				sys_matrix[i][j+offset] = in_matrix[i+interfaceSize+anOffset][j+anOffset]; // 4
				sys_matrix[i][j+offset+interfaceSize] = in_matrix[i+interfaceSize+anOffset][j+offset+interfaceSize+anOffset]; // 6
				sys_matrix[j+offset][i] = in_matrix[j+anOffset][i+interfaceSize+anOffset]; // 2
				sys_matrix[j+offset+interfaceSize][i] = in_matrix[j+offset+interfaceSize+anOffset][i+interfaceSize+anOffset]; // 8
			}
			sys_rhs[i] = in_rhs[i+interfaceSize+anOffset];
		}

		v->system->eliminate(offset);
	}
	else {
		// just copy data?

		for (int i=0; i<leafSize; ++i) {
			for (int j=0; j<leafSize; ++j) {
				sys_matrix[i][j] = in_matrix[i][j];
			}
			sys_rhs[i] = in_rhs[i];
		}

	}
}

void AbstractProduction::A2Node(Vertex *v) const
{
	const int offsetLeft = v->left->type == LEAF ? leafSize-2*interfaceSize : interfaceSize;
	const int offsetRight = v->right->type == LEAF ? leafSize-2*interfaceSize : interfaceSize;

	double** const sys_matrix = v->system->matrix;
	double* const sys_rhs = v->system->rhs;

	const double** const left_matrix = (const double **) v->left->system->matrix;
	const double** const right_matrix = (const double **) v->right->system->matrix;
	const double* const left_rhs = (const double *) v->left->system->rhs;
	const double* const right_rhs = (const double *) v->right->system->rhs;

	for (int i=0; i<this->interfaceSize; ++i) {
		for (int j=0; j<this->interfaceSize; ++j) {
			// x: left y: top
			sys_matrix[i][j] = left_matrix[i+offsetLeft+interfaceSize][j+offsetLeft+interfaceSize] +
					right_matrix[i+offsetRight][j+offsetRight];

			// x: center y: top
			sys_matrix[i][j+interfaceSize] = left_matrix[i+offsetLeft+interfaceSize][j+offsetLeft];

			// x: left y:center
			sys_matrix[i+interfaceSize][j] = left_matrix[i+offsetLeft][j+offsetLeft+interfaceSize];

			// x: center y:center
			sys_matrix[i+interfaceSize][j+interfaceSize] = left_matrix[i+offsetLeft][j+offsetLeft];

			// x: bottom y: bottom
			sys_matrix[i+2*interfaceSize][j+2*interfaceSize] = right_matrix[i+offsetRight+interfaceSize][j+offsetRight+interfaceSize];

			// x: left y:bottom
			sys_matrix[i+2*interfaceSize][j] = right_matrix[i+offsetRight+interfaceSize][j+offsetRight];

			// x: right y: top
			sys_matrix[i][j+2*interfaceSize] = right_matrix[i+offsetRight][j+offsetRight+interfaceSize];
		}
		sys_rhs[i] = left_rhs[i+offsetLeft+interfaceSize] + right_rhs[i+offsetRight];
		sys_rhs[i+interfaceSize] = left_rhs[i+offsetLeft];
		sys_rhs[i+2*interfaceSize] = right_rhs[i+offsetRight + interfaceSize];
	}

	v->system->eliminate(this->interfaceSize);
}

void AbstractProduction::A2Root(Vertex *v) const
{
	const int offsetLeft = v->left->type == LEAF ? leafSize-2*interfaceSize : interfaceSize;
	const int offsetRight = v->right->type == LEAF ? leafSize-2*interfaceSize : interfaceSize;

	double** const sys_matrix = v->system->matrix;
	double* const sys_rhs = v->system->rhs;

	const double** const left_matrix = (const double **) v->left->system->matrix;
	const double** const right_matrix = (const double **) v->right->system->matrix;
	const double* const left_rhs = (const double *) v->left->system->rhs;
	const double* const right_rhs = (const double *) v->right->system->rhs;

	for (int i=0; i<this->interfaceSize; ++i) {
		for (int j=0; j<this->interfaceSize; ++j) {
			// x: left y: top
			sys_matrix[i][j] = left_matrix[i+offsetLeft+interfaceSize][j+offsetLeft+interfaceSize] +
					right_matrix[i+offsetRight][j+offsetRight];

			// x: center y: top
			sys_matrix[i][j+interfaceSize] = left_matrix[i+offsetLeft+interfaceSize][j+offsetLeft];

			// x: left y:center
			sys_matrix[i+interfaceSize][j] = left_matrix[i+offsetLeft][j+offsetLeft+interfaceSize];

			// x: center y:center
			sys_matrix[i+interfaceSize][j+interfaceSize] = left_matrix[i+offsetLeft][j+offsetLeft];

			// x: bottom y: bottom
			sys_matrix[i+2*interfaceSize][j+2*interfaceSize] = right_matrix[i+offsetRight+interfaceSize][j+offsetRight+interfaceSize];

			// x: left y:bottom
			sys_matrix[i+2*interfaceSize][j] = right_matrix[i+offsetRight+interfaceSize][j+offsetRight];

			// x: right y: top
			sys_matrix[i][j+2*interfaceSize] = right_matrix[i+offsetRight][j+offsetRight+interfaceSize];
		}
		sys_rhs[i] = left_rhs[i+offsetLeft+interfaceSize] + right_rhs[i+offsetRight];
		sys_rhs[i+interfaceSize] = left_rhs[i+offsetLeft];
		sys_rhs[i+2*interfaceSize] = right_rhs[i+offsetRight + interfaceSize];
	}

	v->system->eliminate(this->interfaceSize * 3);
}

void AbstractProduction::BS(Vertex *v) const
{
	if (v->type == ROOT) {
		v->system->backwardSubstitute(this->interfaceSize * 3-1);
	}
	else if (v->type == NODE) {
		const int offsetA = v->parent->left == v ? interfaceSize*2 : interfaceSize;
		const int offsetB = v->parent->left == v ? interfaceSize : interfaceSize*2;

		double** const sys_matrix = v->system->matrix;
		double* const sys_rhs = v->system->rhs;

		const double* const par_rhs = (const double *) v->parent->system->rhs;

		for (int i=interfaceSize;i<3*interfaceSize; ++i) {
			for (int j=i; j<3*interfaceSize; ++j) {
				sys_matrix[i][j] = i == j ? 1.0 : 0.0;
			}
		}

		for (int i=0; i<this->interfaceSize; ++i) {
			sys_rhs[i+offsetA] = par_rhs[i];
			sys_rhs[i+offsetB] = par_rhs[i+offsetB];
		}

		v->system->backwardSubstitute(interfaceSize-1);
	}
	else if (v->type == LEAF) {
		const int offsetA = v->parent->left == v ? interfaceSize : 0;
		const int offsetB = v->parent->left == v ? 0 : 2*interfaceSize;

		double** const sys_matrix = v->system->matrix;
		double* const sys_rhs = v->system->rhs;

		const double* const par_rhs = (const double *) v->parent->system->rhs;

		for (int i=leafSize-2*interfaceSize; i<leafSize; ++i) {
			for (int j=i; j<leafSize; ++j) {
				sys_matrix[i][j] = (i==j) ? 1.0 : 0.0;
			}
		}

		for (int i=0; i<this->interfaceSize; ++i) {
			sys_rhs[i+offset] = par_rhs[i+offsetA];
			sys_rhs[i+offset+interfaceSize] = par_rhs[i+offsetB];
		}

		v->system->backwardSubstitute(leafSize-2*interfaceSize);
	}
}

int AbstractProduction::getInterfaceSize() const
{
	return this->interfaceSize;
}

int AbstractProduction::getLeafSize() const
{
	return this->leafSize;
}

int AbstractProduction::getA1Size() const
{
	return this->a1Size;
}

int AbstractProduction::getANSize() const
{
	return this->anSize;
}

