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
				a1->rhs[j+offset+leafOffset] = leafs->at(i)->system->rhs[j+leafOffset+productions->getInterfaceSize()];
			}

			for (j=0;j<leafOffset; ++j) {
				a1->rhs[j+offset+productions->getInterfaceSize()] = leafs->at(i)->system->rhs[j];
			}

			a1->backwardSubstitute(offset);

			for (j=0;j<offset;j+=2) {
				(*result)[j] = a1->rhs[j];
				(*result)[j+1] = a1->rhs[j+offset];
			}

			for (j=0;j<a1->n-2*offset;++j) {
				(*result)[j+2*offset] = a1->rhs[j+2*offset];
			}

		} else if (i == leafs->size()-1) {
			leafOffset = productions->getANSize() - productions->getLeafSize();
			totalOffset = productions->getA1Size() + (i-1)*productions->getLeafSize();
			offset = productions->getANSize() - productions->getLeafSize();

			for (j=0; j<leafOffset-productions->getLeafSize(); ++j) {
				(*result)[j+totalOffset] = leafs->at(i)->system->rhs[j+offset];
			}

			for (j=0; j<productions->getInterfaceSize(); ++j) {
				(*result)[j+totalOffset+leafOffset-productions->getInterfaceSize()] = leafs->at(i)->system->rhs[j+offset+leafOffset];
			}

			for (j=0; j<offset; ++j) {
				(*result)[j+totalOffset+leafOffset] = leafs->at(i)->system->rhs[j];
			}

		} else {
			totalOffset = productions->getA1Size() + (i-1)*productions->getLeafSize();
			leafOffset = productions->getLeafSize() - productions->getInterfaceSize();

			for (j=0; j<productions->getInterfaceSize(); ++j) {
				(*result)[i+totalOffset+productions->getInterfaceSize()] = leafs->at(i)->system->rhs[j+leafOffset];
			}

			for (j=0; j<leafOffset-productions->getInterfaceSize(); ++j) {
				(*result)[i+totalOffset] = leafs->at(i)->system->rhs[j];
			}
		}
	}

	return result;
}
