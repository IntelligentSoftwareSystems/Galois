#include <stdio.h>
//#include "../Productions/GraphGenerator.hxx"
//#include "TripleArgFunction.hxx"
#include "MatrixGenerator.hxx"
#include "../EquationSystem.h"
#include <vector>
#include <stdlib.h>
#include <time.h>
//using namespace D3;
class MyFunction : public D3::ITripleArgFunction
{
	public:
		virtual double ComputeValue(double x, double y, double z)
		{
			return 3*x*x + 2*y*y + z*z + x*y*z + x*x*z + y*y;// + 12*x*x*y*y*z*z;
		}
};

void print_matrix_rhs(double** matrix, double* rhs, int size)
{
	for(int i = 0; i<size; i++){
		printf("%d ",i);
		for(int j = 0; j<size; j++){
				printf("%0.7lf ",matrix[i][j]);
		}

		printf("| %0.7lf\n",rhs[i]);


	}
}

void print_matrix(double** matrix, int size)
{
	for(int i = 0; i<size; i++){
		for(int j = 0; j<size; j++){
				printf("%0.7lf ",matrix[i][j]);
		}
		printf("\n");

	}
}

void print_rhs(double* rhs, int size)
{
	for(int i = 0; i<size; i++){
		printf("%d %0.7lf\n",i,rhs[i]);
	}
}

double sum(int size, int row, int* nrs, double** matrix)
{
	double sum = 0;
	for(int i = 0; i<size ; i++)
		sum+=matrix[row][nrs[i]];
	return sum;
}

int main(int argc, char** argv)
{

	D3::MatrixGenerator* matrix_generator = new D3::MatrixGenerator();
	D3::ITripleArgFunction* my_function = new MyFunction();
	int nr_of_tiers = 3;
	std::list<D3::Tier*>* tier_list = matrix_generator->CreateMatrixAndRhs(nr_of_tiers, 0, 0, 0, 4, my_function);
					  
	double** matrix = matrix_generator->GetMatrix();
	double* rhs = matrix_generator->GetRhs();
	int size = matrix_generator->GetMatrixSize();
	srand(time(NULL));


	EquationSystem* eq = new EquationSystem(matrix,rhs,size);
	eq->eliminate(size);
	eq->backwardSubstitute(size-1);
	//print_rhs(eq->rhs,size);



	std::map<int,double> *map = new std::map<int,double>();
	for(int i = 0; i<181 + 56; i++)
		(*map)[i] = eq->rhs[i];
	matrix_generator->checkSolution(map,my_function);


	return 0;

}

