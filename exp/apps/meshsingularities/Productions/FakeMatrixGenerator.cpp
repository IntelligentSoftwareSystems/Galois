/*
 * FakeMatrixGenerator.cpp
 *
 *  Created on: Aug 28, 2013
 *      Author: kjopek
 */

#include "FakeMatrixGenerator.h"

std::vector<EquationSystem*> *FakeMatrixGenerator::CreateMatrixAndRhs(TaskDescription& taskDescription)
{

	std::vector<EquationSystem*> *leafVector = new std::vector<EquationSystem*>();

	const int iSize = this->getiSize(taskDescription.polynomialDegree);
	const int leafSize = this->getLeafSize(taskDescription.polynomialDegree);
	const int a1Size = this->getA1Size(taskDescription.polynomialDegree);
	const int aNSize = this->getANSize(taskDescription.polynomialDegree);

	for (int i = 0; i < taskDescription.nrOfTiers; ++i) {
		int n;
		if (i==0) {
			n = a1Size;
		}
		else if (i==taskDescription.nrOfTiers-1) {
			n = aNSize;
		}
		else {
			n = leafSize + 2*iSize;
		}

		EquationSystem *system = new EquationSystem(n);
		for (int j=0; j<n; ++j) {
			for (int k=0; k<n; ++k) {
				system->matrix[j][k] = (INT_MAX-rand())/INT_MAX*1.0;
			}
			system->rhs[j] = 1.0;
		}

		leafVector->push_back(system);
	}

	return leafVector;
}

void FakeMatrixGenerator::checkSolution(std::map<int,double> *solution_map, double (*f)(int dim, ...))
{
 // empty
}

