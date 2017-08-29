#include <stdio.h>
#include "MatrixGenerator.hxx"
#include <map>
#include <time.h>
#include <map>
#include "../EquationSystem.h"

using namespace D2;

double test_function(int dim, ...)
{
	double *data = new double[dim];
	double result = 0;
	va_list args;

	va_start (args, dim);
	for (int i=0; i<dim; ++i) {
		data[i] = va_arg (args, double);
	}
	va_end(args);

	if (dim == 2)
	{


		result = data[0]*data[1]+data[0]+data[0]*data[1]*data[0]*data[1] + 13;
	}
	else
	{
		result = -1;
	}

	delete [] data;
	return result;
}

int main(int argc, char** argv)
{


		GenericMatrixGenerator *matrixGenerator = new MatrixGenerator();

		TaskDescription taskDescription;
		taskDescription.dimensions=2;
		taskDescription.nrOfTiers=10;
		taskDescription.size=2;
		taskDescription.function=test_function;
		taskDescription.x=-1;
		taskDescription.y=-1;
		std::vector<EquationSystem*> *tiers = matrixGenerator->CreateMatrixAndRhs(taskDescription);
		EquationSystem *globalSystem = new EquationSystem(matrixGenerator->GetMatrix(),
														  matrixGenerator->GetRhs(),
														  matrixGenerator->GetMatrixSize());

		//globalSystem->print();
		globalSystem->eliminate(matrixGenerator->GetMatrixSize());
		globalSystem->backwardSubstitute(matrixGenerator->GetMatrixSize()-1);

		std::map<int,double>* result_map = new std::map<int,double>();
		for(int i = 0; i<matrixGenerator->GetMatrixSize(); i++)
		{
			(*result_map)[i] = globalSystem->rhs[i];
		}

		matrixGenerator->checkSolution(result_map,test_function);

		delete matrixGenerator;
		delete result_map;
		return 0;
}

