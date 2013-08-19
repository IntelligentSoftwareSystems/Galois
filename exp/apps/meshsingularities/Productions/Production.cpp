#include "Production.h"

void AbstractProduction::A1(Vertex *v, EquationSystem *inData)
{
	// here we assumed, that inData is currently pre-processed

	int i, j;
	int offset = leafSize - 2*interfaceSize;

	int a1Offset = a1Size - leafSize;


	if (offset > 0) {
		for (i=0; i<interfaceSize; i++) {
			for (j=0; j<interfaceSize; j++) {
				v->system->matrix[i+offset][j+offset] = inData->matrix[i+a1Offset][j+a1Offset]; // 1
				v->system->matrix[i+offset][j+offset+interfaceSize] = inData->matrix[i+a1Offset][j+offset+interfaceSize+a1Offset]; // 3
				v->system->matrix[i+offset+interfaceSize][j+offset] = inData->matrix[i+offset+interfaceSize+a1Offset][j+a1Offset]; // 7
				v->system->matrix[i+offset+interfaceSize][j+offset+interfaceSize] = inData->matrix[i+offset+interfaceSize+a1Offset][j+offset+interfaceSize+a1Offset]; // 9
			}
		}

		for (i=0; i<offset; i++) {
			for (j=0; j<offset; j++) {
				v->system->matrix[i][j] = inData->matrix[i+interfaceSize+a1Offset][j+interfaceSize+a1Offset]; // 5
			}
		}

		for (i=0; i<offset; i++) {
			for (j=0; j<interfaceSize; j++) {
				v->system->matrix[i][j+offset] = inData->matrix[i+interfaceSize+a1Offset][j+a1Offset]; // 4
				v->system->matrix[i][j+offset+interfaceSize] = inData->matrix[i+interfaceSize+a1Offset][j+offset+interfaceSize+a1Offset]; // 6
				v->system->matrix[j+offset][i] = inData->matrix[j+a1Offset][i+interfaceSize+a1Offset]; // 2
				v->system->matrix[j+offset+interfaceSize][i] = inData->matrix[j+offset+interfaceSize+a1Offset][i+interfaceSize+a1Offset]; // 8
			}
		}

		for (i=0; i<interfaceSize; i++) {
			v->system->rhs[i+offset] = inData->rhs[i+a1Offset];
			v->system->rhs[i+offset+interfaceSize] = inData->rhs[i+offset+interfaceSize+a1Offset];
		}

		for (int i=0; i<offset; i++) {
			v->system->rhs[i] = inData->rhs[i+interfaceSize+a1Offset];
		}

		v->system->eliminate(leafSize-2*interfaceSize);
	}
	else {
		// just copy data?

		for (i=0; i<leafSize; ++i) {
			for (j=0; j<leafSize; ++j) {
				v->system->matrix[i][j] = inData->matrix[i][j];
			}
			v->system->rhs[i] = inData->rhs[i];
		}

	}
}

void AbstractProduction::A(Vertex *v, EquationSystem *inData)
{
	// You will probably need to overwrite this method
	// to adjust it to your requirements.

	int i,j;
	int offset = leafSize - 2*interfaceSize;

	if (offset > 0) {
		for (i=0; i<interfaceSize; i++) {
			for (j=0; j<interfaceSize; j++) {
				v->system->matrix[i+offset][j+offset] = inData->matrix[i][j]; // 1
				v->system->matrix[i+offset][j+offset+interfaceSize] = inData->matrix[i][j+offset+interfaceSize]; // 3
				v->system->matrix[i+offset+interfaceSize][j+offset] = inData->matrix[i+offset+interfaceSize][j]; // 7
				v->system->matrix[i+offset+interfaceSize][j+offset+interfaceSize] = inData->matrix[i+offset+interfaceSize][j+offset+interfaceSize]; // 9
			}
		}

		for (i=0; i<offset; i++) {
			for (j=0; j<offset; j++) {
				v->system->matrix[i][j] = inData->matrix[i+interfaceSize][j+interfaceSize]; // 5
			}
		}

		for (i=0; i<offset; i++) {
			for (j=0; j<interfaceSize; j++) {
				v->system->matrix[i][j+offset] = inData->matrix[i+interfaceSize][j]; // 4
				v->system->matrix[i][j+offset+interfaceSize] = inData->matrix[i+interfaceSize][j+offset+interfaceSize]; // 6
				v->system->matrix[j+offset][i] = inData->matrix[j][i+interfaceSize]; // 2
				v->system->matrix[j+offset+interfaceSize][i] = inData->matrix[j+offset+interfaceSize][i+interfaceSize]; // 8
			}
		}

		for (i=0; i<interfaceSize; i++) {
			v->system->rhs[i+offset] = inData->rhs[i];
			v->system->rhs[i+offset+interfaceSize] = inData->rhs[i+offset+interfaceSize];
		}

		for (int i=0; i<offset; i++) {
			v->system->rhs[i] = inData->rhs[i+interfaceSize];
		}

		v->system->eliminate(offset);

	}
	else {
		for (i=0; i<interfaceSize*2; ++i) {
			for (j=0; j<interfaceSize*2; ++j) {
				v->system->matrix[i][j] = inData->matrix[i][j];
			}
			v->system->rhs[i] = inData->rhs[i];
		}
	}

}

void AbstractProduction::AN(Vertex *v, EquationSystem *inData)
{
	// identical to A1

	int i, j;
	int offset = leafSize - 2*interfaceSize;

	int anOffset = anSize - leafSize;
	if (offset > 0) {
		for (i=0; i<interfaceSize; i++) {
			for (j=0; j<interfaceSize; j++) {
				v->system->matrix[i+offset][j+offset] = inData->matrix[i+anOffset][j+anOffset]; // 1
				v->system->matrix[i+offset][j+offset+interfaceSize] = inData->matrix[i+anOffset][j+offset+interfaceSize+anOffset]; // 3
				v->system->matrix[i+offset+interfaceSize][j+offset] = inData->matrix[i+offset+interfaceSize+anOffset][j+anOffset]; // 7
				v->system->matrix[i+offset+interfaceSize][j+offset+interfaceSize] = inData->matrix[i+offset+interfaceSize+anOffset][j+offset+interfaceSize+anOffset]; // 9
			}
		}

		for (i=0; i<offset; i++) {
			for (j=0; j<offset; j++) {
				v->system->matrix[i][j] = inData->matrix[i+interfaceSize+anOffset][j+interfaceSize+anOffset]; // 5
			}
		}

		for (i=0; i<offset; i++) {
			for (j=0; j<interfaceSize; j++) {
				v->system->matrix[i][j+offset] = inData->matrix[i+interfaceSize+anOffset][j+anOffset]; // 4
				v->system->matrix[i][j+offset+interfaceSize] = inData->matrix[i+interfaceSize+anOffset][j+offset+interfaceSize+anOffset]; // 6
				v->system->matrix[j+offset][i] = inData->matrix[j+anOffset][i+interfaceSize+anOffset]; // 2
				v->system->matrix[j+offset+interfaceSize][i] = inData->matrix[j+offset+interfaceSize+anOffset][i+interfaceSize+anOffset]; // 8
			}
		}

		for (i=0; i<interfaceSize; i++) {
			v->system->rhs[i+offset] = inData->rhs[i+anOffset];
			v->system->rhs[i+offset+interfaceSize] = inData->rhs[i+offset+interfaceSize+anOffset];
		}

		for (int i=0; i<offset; i++) {
			v->system->rhs[i] = inData->rhs[i+interfaceSize+anOffset];
		}

		v->system->eliminate(leafSize-2*interfaceSize);
	}
	else {
		// just copy data?

		for (i=0; i<leafSize; ++i) {
			for (j=0; j<leafSize; ++j) {
				v->system->matrix[i][j] = inData->matrix[i][j];
			}
			v->system->rhs[i] = inData->rhs[i];
		}

	}
}

void AbstractProduction::A2(Vertex *v)
{
	int i, j;
	int offsetLeft = v->left->type == LEAF ? leafSize-2*interfaceSize : interfaceSize;
	int offsetRight = v->right->type == LEAF ? leafSize-2*interfaceSize : interfaceSize;


	for (i=0; i<this->interfaceSize; ++i) {
		for (j=0; j<this->interfaceSize; ++j) {
			// x: left y: top
			v->system->matrix[i][j] = v->left->system->matrix[i+offsetLeft+interfaceSize][j+offsetLeft+interfaceSize] +
					v->right->system->matrix[i+offsetRight][j+offsetRight];

			// x: center y: top
			v->system->matrix[i][j+interfaceSize] = v->left->system->matrix[i+offsetLeft+interfaceSize][j+offsetLeft];

			// x: left y:center
			v->system->matrix[i+interfaceSize][j] = v->left->system->matrix[i+offsetLeft][j+offsetLeft+interfaceSize];

			// x: center y:center
			v->system->matrix[i+interfaceSize][j+interfaceSize] = v->left->system->matrix[i+offsetLeft][j+offsetLeft];

			// x: bottom y: bottom
			v->system->matrix[i+2*interfaceSize][j+2*interfaceSize] = v->right->system->matrix[i+offsetRight+interfaceSize][j+offsetRight+interfaceSize];

			// x: left y:bottom
			v->system->matrix[i+2*interfaceSize][j] = v->right->system->matrix[i+offsetRight+interfaceSize][j+offsetRight];

			// x: right y: top
			v->system->matrix[i][j+2*interfaceSize] = v->right->system->matrix[i+offsetRight][j+offsetRight+interfaceSize];
		}
		v->system->rhs[i] = v->left->system->rhs[i+offsetLeft+interfaceSize] + v->right->system->rhs[i+offsetRight];
		v->system->rhs[i+interfaceSize] = v->left->system->rhs[i+offsetLeft];
		v->system->rhs[i+2*interfaceSize] = v->right->system->rhs[i+offsetRight + interfaceSize];
	}
}

void AbstractProduction::E(Vertex *v)
{
	v->system->eliminate(this->interfaceSize);
}

void AbstractProduction::ERoot(Vertex *v)
{
	v->system->eliminate(this->interfaceSize * 3);
}

void AbstractProduction::BS(Vertex *v)
{
	int i,j;

	if (v->type == ROOT) {
		v->system->backwardSubstitute(this->interfaceSize * 3-1);
		//v->system->print();
	}

	if (v->type == NODE) {
		int offsetA, offsetB;
		offsetA = v->parent->left == v ? interfaceSize*2 : interfaceSize;
		offsetB = v->parent->left == v ? interfaceSize : interfaceSize*2;

		for (i=interfaceSize;i<3*interfaceSize; ++i) {
			for (j=interfaceSize; j<3*interfaceSize; j++) {
				v->system->matrix[i][j] = i == j ? 1.0 : 0.0;
			}
		}

		for (i=0; i<this->interfaceSize; ++i) {
			v->system->rhs[i+offsetA] = v->parent->system->rhs[i];
			v->system->rhs[i+offsetB] = v->parent->system->rhs[i+offsetB];
		}

		v->system->backwardSubstitute(interfaceSize-1);
	}

	if (v->type == LEAF) {
		int offsetA = v->parent->left == v ? interfaceSize : 0;
		int offsetB = v->parent->left == v ? 0 : 2*interfaceSize;
		int offset = leafSize - 2*interfaceSize;

		for (i=leafSize-2*interfaceSize; i<leafSize; ++i) {
			for (j=leafSize-2*interfaceSize; j<leafSize; ++j) {
				v->system->matrix[i][j] = (i==j) ? 1.0 : 0.0;
			}
		}

		for (i=0; i<this->interfaceSize; ++i) {
			v->system->rhs[i+offset] = v->parent->system->rhs[i+offsetA];
			v->system->rhs[i+offset+interfaceSize] = v->parent->system->rhs[i+offsetB];
		}

		v->system->backwardSubstitute(leafSize-2*interfaceSize);
		//if (v->parent->left == v && v->parent->parent->left == v->parent) 
		//	v->system->print();
	}
}

int AbstractProduction::getInterfaceSize() {
	return this->interfaceSize;
}

int AbstractProduction::getLeafSize() {
	return this->leafSize;
}

int AbstractProduction::getA1Size() {
	return this->a1Size;
}

int AbstractProduction::getANSize() {
	return this->anSize;
}

