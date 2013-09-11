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
		AbstractProduction *productions) const
{
	std::vector<EquationSystem *> *outputVector = new std::vector<EquationSystem*>();
	std::list<EquationSystem*>::iterator it = input->begin();

	for (int i = 0; it != input->end(); ++it, ++i) {
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
		AbstractProduction *productions) const
{
	std::vector<double> * result = new std::vector<double>(
			(leafs->size()-2)*productions->getLeafSize()+
			productions->getA1Size()+
			productions->getANSize()-
			(leafs->size()-1)*productions->getInterfaceSize());

	for (int i=0; i<leafs->size(); ++i) {
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


EquationSystem *Processing::preprocessA1(EquationSystem *input, AbstractProduction *productions) const
{
	EquationSystem *system = new EquationSystem(input->matrix, input->rhs, input->n);

	system->eliminate(productions->getA1Size()-productions->getLeafSize());

	return (system);
}

EquationSystem *Processing::preprocessA(EquationSystem *input, AbstractProduction *productions) const
{
	return new EquationSystem(input->matrix, input->rhs, input->n);
}

EquationSystem *Processing::preprocessAN(EquationSystem *input, AbstractProduction *productions) const
{
	EquationSystem *system = new EquationSystem(productions->getANSize());
	double ** const tierMatrix = input->matrix;
	double * const tierRhs = input->rhs;

	double ** const sys_matrix = system->matrix;
	double * const sys_rhs = system->rhs;

	const int leafSize = productions->getLeafSize();
	const int offset = productions->getANSize() - leafSize;
	if (offset > 0) {
		for (int i=0; i<leafSize; ++i) {
			for (int j=0; j<leafSize; ++j) {
				sys_matrix[i+offset][j+offset] = tierMatrix[i][j];
			}
			sys_rhs[i+offset] = tierRhs[i];
		}

		for (int i=0;i<offset;++i) {
			for (int j=0;j<leafSize;++j) {
				sys_matrix[i][j+offset] = tierMatrix[i+leafSize][j];
				sys_matrix[j+offset][i] = tierMatrix[j][i+leafSize];
			}
		}

		for (int i=0; i<offset; ++i) {
			for (int j=0; j<offset; ++j) {
				sys_matrix[i][j] = tierMatrix[i+leafSize][j+leafSize];
			}
			sys_rhs[i] = tierRhs[i+leafSize];
		}

		system->eliminate(offset);
	}
	return (system);
}

void Processing::postprocessA1(Vertex *leaf, EquationSystem *inputData,
		AbstractProduction *productions, std::vector<double> *result, int num) const
{
	EquationSystem *a1 = inputData;

	int offset = productions->getA1Size() - productions->getLeafSize();
	int leafOffset = productions->getLeafSize() - 2*productions->getInterfaceSize();

	if (offset > 0) {
		for (int j=0; j<productions->getLeafSize(); ++j) {
			for (int k=0; k<productions->getLeafSize(); ++k) {
				a1->matrix[j+offset][k+offset] = j==k ? 1.0 : 0.0;
			}
		}

		for (int j=0; j<productions->getInterfaceSize(); ++j) {
			a1->rhs[j+offset] = leaf->system->rhs[j+leafOffset];
			a1->rhs[j+offset+leafOffset+productions->getInterfaceSize()] =
					leaf->system->rhs[j+leafOffset+productions->getInterfaceSize()];
		}

		for (int j=0;j<leafOffset; ++j) {
			a1->rhs[j+offset+productions->getInterfaceSize()] = leaf->system->rhs[j];
		}

		a1->backwardSubstitute(offset);
		for (int j=0;j<productions->getA1Size(); ++j) {
			(*result)[j] = a1->rhs[j];
		}
	}
}

void Processing::postprocessA(Vertex *leaf, EquationSystem *inputData,
		AbstractProduction *productions, std::vector<double> *result, int num) const
{
	const int leafOffset = productions->getLeafSize() - 2*productions->getInterfaceSize();
	const int totalOffset = (productions->getA1Size()-productions->getInterfaceSize())+(num-1)*(productions->getLeafSize()-productions->getInterfaceSize());

	for (int j=0; j<productions->getInterfaceSize(); ++j) {
		(*result)[totalOffset+j+leafOffset+productions->getInterfaceSize()] = leaf->system->rhs[j+leafOffset+productions->getInterfaceSize()];
	}

	for (int j=0; j<leafOffset; ++j) {
		(*result)[totalOffset+j+productions->getInterfaceSize()] = leaf->system->rhs[j];
	}
}

void Processing::postprocessAN(Vertex *leaf, EquationSystem *inputData,
		AbstractProduction *productions, std::vector<double> *result, int num) const
{

	const int leafOffset = productions->getLeafSize() - 2*productions->getInterfaceSize();
	const int totalOffset = (productions->getA1Size()-productions->getInterfaceSize())+(num-1)*(productions->getLeafSize()-productions->getInterfaceSize());
	const int offset = productions->getANSize() - productions->getLeafSize();

	EquationSystem *an = inputData;

	for (int j=0; j<productions->getLeafSize(); ++j) {
		for (int k=0; k<productions->getLeafSize(); ++k) {
			an->matrix[j+offset][k+offset] = j==k ? 1.0 : 0.0;
		}
	}

	for (int j=0; j<productions->getInterfaceSize(); ++j) {
		an->rhs[j+offset] = leaf->system->rhs[j+leafOffset];
		an->rhs[j+offset+leafOffset+productions->getInterfaceSize()] = leaf->system->rhs[j+leafOffset+productions->getInterfaceSize()];
	}

	for (int j=0;j<leafOffset; ++j) {
		an->rhs[j+offset+productions->getInterfaceSize()] = leaf->system->rhs[j];
	}

	an->backwardSubstitute(offset);

	for (int j=0; j<productions->getLeafSize(); ++j) {
		(*result)[j+totalOffset] = an->rhs[j+offset];
	}

	for (int j=0; j<offset; ++j) {
		(*result)[j+totalOffset+productions->getLeafSize()] = an->rhs[j];
	}

}

std::vector<Vertex*> *Processing::collectLeafs(Vertex *p)
{
	std::vector<Vertex *> *left = NULL;
	std::vector<Vertex *> *right = NULL;
	std::vector<Vertex*> *result = NULL;

	if (p == NULL) {
		return NULL;
	}

	result = new std::vector<Vertex*>();

	if (p!=NULL && p->right==NULL && p->left==NULL) {
		result->push_back(p);
		return result;
	}

	if (p!=NULL && p->left!=NULL) {
		left = collectLeafs(p->left);
	}

	if (p!=NULL && p->right!=NULL) {
		right = collectLeafs(p->right);
	}

	if (left != NULL) {
		for (std::vector<Vertex*>::iterator it = left->begin(); it!=left->end(); ++it) {
			if (*it != NULL) {
				result->push_back(*it);
			}
		}
		delete left;
	}

	if (right != NULL) {
		for (std::vector<Vertex*>::iterator it = right->begin(); it!=right->end(); ++it) {
			if (*it != NULL) {
				result->push_back(*it);
			}
		}
		delete right;
	}

	return result;
}
