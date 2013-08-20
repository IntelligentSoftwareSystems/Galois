/*
 * Processing.cpp
 *
 *  Created on: Aug 20, 2013
 *      Author: kjopek
 */

#include "Processing.h"

Processing::Processing() {

}

Processing::~Processing() {
}

std::vector<EquationSystem*>* Processing::preprocess(std::list<EquationSystem*>* input,
		AbstractProduction *productions)
{
	std::vector<EquationSystem *> *outputVector = new std::vector<EquationSystem*>();
	int i = 0;
	std::list<EquationSystem*>::iterator it = input->begin();

	for (; it != input->end(); ++it, ++i) {
		if (i==0) {
			outputVector->push_back(this->preprocessA1(*it, productions));
		} else if (i==input->size()-1) {
			outputVector->push_back(this->preprocessAN(*it, productions));
		} else {
			outputVector->push_back(this->preprocessA(*it, productions));
		}
	}
	return outputVector;
}

std::vector<double> *Processing::postprocess(std::vector<Vertex *> *leafs,
		std::vector<EquationSystem *> *inputData,
		AbstractProduction *productions)
{
	int i=0;
	const int eqnSize = (leafs->size()-2)*productions->getLeafSize()+
			productions->getA1Size()+
			productions->getANSize()-
			(leafs->size()-1)*productions->getInterfaceSize();

	std::vector<double> * result = new std::vector<double>(eqnSize);

	for (i=0; i<leafs->size(); ++i) {
		if (i==0) {
			this->postprocessA1(leafs->at(i), inputData->at(i), productions, result, i);
		} else if (i == leafs->size()-1) {
			this->postprocessAN(leafs->at(i), inputData->at(i), productions, result, i);
		} else {
			this->postprocessA(leafs->at(i), inputData->at(i), productions, result, i);
		}
	}

	return result;
}


EquationSystem *Processing::preprocessA1(EquationSystem *input, AbstractProduction *productions)
{
	EquationSystem *system = new EquationSystem(input->matrix, input->rhs, input->n);

	system->eliminate(productions->getA1Size()-productions->getLeafSize());

	return (system);
}

EquationSystem *Processing::preprocessA(EquationSystem *input, AbstractProduction *productions)
{
	EquationSystem *system = new EquationSystem(input->matrix, input->rhs, input->n);
	return (system);
}

EquationSystem *Processing::preprocessAN(EquationSystem *input, AbstractProduction *productions)
{
	EquationSystem *system = new EquationSystem(productions->getANSize());
	double ** tierMatrix = input->matrix;
	double *tierRhs = input->rhs;

	int leafSize = productions->getLeafSize();
	int offset = productions->getANSize() - leafSize;

	for (int i=0; i<leafSize; i++) {
		for (int j=0; j<leafSize; j++) {
			system->matrix[i+offset][j+offset] = tierMatrix[i][j];
		}
		system->rhs[i+offset] = tierRhs[i];
	}

	for (int i=0;i<offset;i++) {
		for (int j=0;j<leafSize;j++) {
			system->matrix[i][j+offset] = tierMatrix[i+leafSize][j];
			system->matrix[j+offset][i] = tierMatrix[j][i+leafSize];
		}
	}

	for (int i=0; i<offset; i++) {
		for (int j=0; j<offset; j++) {
			system->matrix[i][j] = tierMatrix[i+leafSize][j+leafSize];
		}
		system->rhs[i] = tierRhs[i+leafSize];
	}

	system->eliminate(offset);

	return (system);
}

void Processing::postprocessA1(Vertex *leaf, EquationSystem *inputData,
		AbstractProduction *productions, std::vector<double> *result, int num)
{
	EquationSystem *a1 = inputData;

	int i,j,k;
	i = j = k = 0;

	int offset = productions->getA1Size() - productions->getLeafSize();
	int leafOffset = productions->getLeafSize() - 2*productions->getInterfaceSize();

	for (j=0; j<productions->getLeafSize(); ++j) {
		for (k=0; k<productions->getLeafSize(); ++k) {
			a1->matrix[j+offset][k+offset] = j==k ? 1.0 : 0.0;
		}
	}

	for (j=0; j<productions->getInterfaceSize(); j++) {
		a1->rhs[j+offset] = leaf->system->rhs[j+leafOffset];
		a1->rhs[j+offset+leafOffset+productions->getInterfaceSize()] =
				leaf->system->rhs[j+leafOffset+productions->getInterfaceSize()];
	}

	for (j=0;j<leafOffset; ++j) {
		a1->rhs[j+offset+productions->getInterfaceSize()] = leaf->system->rhs[j];
	}

	a1->backwardSubstitute(offset);
	for (j=0;j<productions->getA1Size(); ++j) {
		(*result)[j] = a1->rhs[j];
	}
}

void Processing::postprocessA(Vertex *leaf, EquationSystem *inputData,
		AbstractProduction *productions, std::vector<double> *result, int num)
{
	int j = 0;
	int leafOffset = productions->getLeafSize() - 2*productions->getInterfaceSize();
	int totalOffset = (productions->getA1Size()-productions->getInterfaceSize())+(num-1)*(productions->getLeafSize()-productions->getInterfaceSize());

	for (j=0; j<productions->getInterfaceSize(); ++j) {
		(*result)[totalOffset+j+leafOffset+productions->getInterfaceSize()] = leaf->system->rhs[j+leafOffset+productions->getInterfaceSize()];
	}

	for (j=0; j<leafOffset; ++j) {
		(*result)[totalOffset+j+productions->getInterfaceSize()] = leaf->system->rhs[j];
	}
}

void Processing::postprocessAN(Vertex *leaf, EquationSystem *inputData,
		AbstractProduction *productions, std::vector<double> *result, int num)
{

	int j, k;
	j = k = 0;

	int leafOffset = productions->getLeafSize() - 2*productions->getInterfaceSize();
	int totalOffset = (productions->getA1Size()-productions->getInterfaceSize())+(num-1)*(productions->getLeafSize()-productions->getInterfaceSize());
	int offset = productions->getANSize() - productions->getLeafSize();

	EquationSystem *an = inputData;

	for (j=0; j<productions->getLeafSize(); ++j) {
		for (k=0; k<productions->getLeafSize(); ++k) {
			an->matrix[j+offset][k+offset] = j==k ? 1.0 : 0.0;
		}
	}

	for (j=0; j<productions->getInterfaceSize(); ++j) {
		an->rhs[j+offset] = leaf->system->rhs[j+leafOffset];
		an->rhs[j+offset+leafOffset+productions->getInterfaceSize()] = leaf->system->rhs[j+leafOffset+productions->getInterfaceSize()];
	}

	for (j=0;j<leafOffset; ++j) {
		an->rhs[j+offset+productions->getInterfaceSize()] = leaf->system->rhs[j];
	}

	an->backwardSubstitute(offset);

	for (j=0; j<productions->getLeafSize(); ++j) {
		(*result)[j+totalOffset] = an->rhs[j+offset];
	}

	for (j=0; j<offset; ++j) {
		(*result)[j+totalOffset+productions->getLeafSize()] = an->rhs[j];
	}

}
