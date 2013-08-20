/*
 * Postprocessor.cpp
 *
 *  Created on: 14-08-2013
 *      Author: kj
 */

#include "Postprocessor.h"

Postprocessor::~Postprocessor() {
}

std::vector<double> *Postprocessor::postprocess(std::vector<Vertex *> *leafs,
		std::vector<EquationSystem *> *inputData,
		AbstractProduction *productions)
{
	int i=0, j=0, k=0;
	int offset, leafOffset, totalOffset;
	const int eqnSize = (leafs->size()-2)*productions->getLeafSize()+
			productions->getA1Size()+
			productions->getANSize()-
			(leafs->size()-1)*productions->getInterfaceSize() + 1;

	std::vector<double> * result = new std::vector<double>(eqnSize);

	for (i=0; i<leafs->size(); ++i) {
		if (i==0) {
			// A1
			EquationSystem *a1 = inputData->at(0);

			offset = productions->getA1Size() - productions->getLeafSize();
			leafOffset = productions->getLeafSize() - 2*productions->getInterfaceSize();

			for (j=0; j<productions->getLeafSize(); ++j) {
				for (k=0; k<productions->getLeafSize(); ++k) {
					a1->matrix[j+offset][k+offset] = j==k ? 1.0 : 0.0;
				}
			}

			for (j=0; j<productions->getInterfaceSize(); j++) {
				a1->rhs[j+offset] = leafs->at(i)->system->rhs[j+leafOffset];
				a1->rhs[j+offset+leafOffset+productions->getInterfaceSize()] =
						leafs->at(i)->system->rhs[j+leafOffset+productions->getInterfaceSize()];
			}

			for (j=0;j<leafOffset; ++j) {
				a1->rhs[j+offset+productions->getInterfaceSize()] = leafs->at(i)->system->rhs[j];
			}


			a1->backwardSubstitute(offset);

			for (j=0;j<offset;++j) {
				(*result)[2*j] = a1->rhs[j+offset];
				(*result)[2*j+1] = a1->rhs[j];
			}

			for (j=0;j<a1->n-2*offset;++j) {
				(*result)[j+2*offset] = a1->rhs[j+2*offset];
			}

		} else if (i == leafs->size()-1) {
			// AN
			leafOffset = productions->getLeafSize()-2*productions->getInterfaceSize();
			totalOffset = (productions->getA1Size()-productions->getInterfaceSize())+(i-1)*(productions->getLeafSize()-productions->getInterfaceSize());
			offset = productions->getANSize() - productions->getLeafSize();

			EquationSystem *an = inputData->at(leafs->size()-1);

			for (j=0; j<productions->getLeafSize(); ++j) {
				for (k=0; k<productions->getLeafSize(); ++k) {
					an->matrix[j+offset][k+offset] = j==k ? 1.0 : 0.0;
				}
			}

			for (j=0; j<productions->getInterfaceSize(); j++) {
				an->rhs[j+offset] = leafs->at(i)->system->rhs[j+leafOffset];
				an->rhs[j+offset+leafOffset+productions->getInterfaceSize()] =
						leafs->at(i)->system->rhs[j+leafOffset+productions->getInterfaceSize()];
			}

			for (j=0;j<leafOffset; ++j) {
				an->rhs[j+offset+productions->getInterfaceSize()] = leafs->at(i)->system->rhs[j];
			}


			an->backwardSubstitute(offset);

			for (j=0; j<productions->getLeafSize(); ++j) {
				(*result)[j+totalOffset] = an->rhs[j+offset];
				printf("setting #%d\n", j+totalOffset);
			}

			for (j=0; j<offset; ++j) {
				(*result)[j+totalOffset+productions->getLeafSize()] = an->rhs[j];
				printf("setting #%d\n", j+totalOffset+productions->getLeafSize());
			}

		} else {
			// A
			leafOffset = productions->getLeafSize() - 2*productions->getInterfaceSize();
			totalOffset = (productions->getA1Size()-productions->getInterfaceSize())+(i-1)*(productions->getLeafSize()-productions->getInterfaceSize());
			for (j=0; j<productions->getInterfaceSize(); ++j) {
				(*result)[totalOffset+j] = leafs->at(i)->system->rhs[j+leafOffset];
				printf("setting #%d\n", j+totalOffset);
				(*result)[totalOffset+j+leafOffset+productions->getInterfaceSize()] = leafs->at(i)->system->rhs[j+leafOffset+productions->getInterfaceSize()];
				printf("setting #%d\n", totalOffset+j+leafOffset+productions->getInterfaceSize());
			}

			for (j=0; j<leafOffset; ++j) {
				(*result)[totalOffset+j+productions->getInterfaceSize()] = leafs->at(i)->system->rhs[j];
				printf("setting #%d\n", totalOffset+j+productions->getInterfaceSize());
			}
		}
	}

	return result;
}

std::vector<double> *Postprocessor3D::postprocess(std::vector<Vertex *> *leafs,
		std::vector<EquationSystem *> *inputData,
		AbstractProduction *productions)
{
	int i=0, j=0, k=0;
	int offset, leafOffset, totalOffset;
	const int eqnSize = (leafs->size()-2)*productions->getLeafSize()+
			productions->getA1Size()+
			productions->getANSize()-
			(leafs->size()-1)*productions->getInterfaceSize() + 1;

	std::vector<double> * result = new std::vector<double>(eqnSize);

	for (i=0; i<leafs->size(); ++i) {
		if (i==0) {
			EquationSystem *a1 = inputData->at(0);

			offset = productions->getA1Size() - productions->getLeafSize();
			leafOffset = productions->getLeafSize() - 2*productions->getInterfaceSize();

			for (j=0; j<productions->getLeafSize(); ++j) {
				for (k=0; k<productions->getLeafSize(); ++k) {
					a1->matrix[j+offset][k+offset] = j==k ? 1.0 : 0.0;
				}
			}

			for (j=0; j<productions->getInterfaceSize(); j++) {
				a1->rhs[j+offset] = leafs->at(i)->system->rhs[j+leafOffset];
				a1->rhs[j+offset+leafOffset+productions->getInterfaceSize()] =
						leafs->at(i)->system->rhs[j+leafOffset+productions->getInterfaceSize()];
			}

			for (j=0;j<leafOffset; ++j) {
				a1->rhs[j+offset+productions->getInterfaceSize()] = leafs->at(i)->system->rhs[j];
			}

			a1->backwardSubstitute(offset);
			for (j=0;j<productions->getA1Size(); ++j) {
				(*result)[j] = a1->rhs[j];
			}

		} else if (i == leafs->size()-1) {
			leafOffset = productions->getLeafSize() - 2*productions->getInterfaceSize();
			totalOffset = (productions->getA1Size()-productions->getInterfaceSize())+(i-1)*(productions->getLeafSize()-productions->getInterfaceSize());
			offset = productions->getANSize() - productions->getLeafSize();

			EquationSystem *an = inputData->at(leafs->size()-1);

			for (j=0; j<productions->getLeafSize(); ++j) {
				for (k=0; k<productions->getLeafSize(); ++k) {
					an->matrix[j+offset][k+offset] = j==k ? 1.0 : 0.0;
				}
			}

			for (j=0; j<productions->getInterfaceSize(); ++j) {
				an->rhs[j+offset] = leafs->at(i)->system->rhs[j+leafOffset];
				an->rhs[j+offset+leafOffset+productions->getInterfaceSize()] = leafs->at(i)->system->rhs[j+leafOffset+productions->getInterfaceSize()];
			}

			for (j=0;j<leafOffset; ++j) {
				an->rhs[j+offset+productions->getInterfaceSize()] = leafs->at(i)->system->rhs[j];
			}

			an->backwardSubstitute(offset);

			an->print();

			for (j=0; j<productions->getLeafSize(); ++j) {
				(*result)[j+totalOffset] = an->rhs[j+offset];
			}

			for (j=0; j<offset; ++j) {
				(*result)[j+totalOffset+productions->getLeafSize()] = an->rhs[j];
			}


		} else {
			// A
			leafOffset = productions->getLeafSize() - 2*productions->getInterfaceSize();
			totalOffset = (productions->getA1Size()-productions->getInterfaceSize())+(i-1)*(productions->getLeafSize()-productions->getInterfaceSize());
			for (j=0; j<productions->getInterfaceSize(); ++j) {
				(*result)[totalOffset+j+leafOffset+productions->getInterfaceSize()] = leafs->at(i)->system->rhs[j+leafOffset+productions->getInterfaceSize()];
			}

			for (j=0; j<leafOffset; ++j) {
				(*result)[totalOffset+j+productions->getInterfaceSize()] = leafs->at(i)->system->rhs[j];
			}
		}
	}


	return result;
}
