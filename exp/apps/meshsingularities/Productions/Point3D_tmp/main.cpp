#include <stdio.h>
//#include "../Productions/GraphGenerator.hxx"
//#include "TripleArgFunction.hxx"
#include "MatrixGenerator.hxx"
//#include "../EquationSystem.h"
#include <vector>
using namespace tmp;
class MyFunction : public ITripleArgFunction
{
	public:
		virtual double ComputeValue(double x, double y, double z)
		{
			return x*x + y + z;
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
		printf("%0.7lf\n",rhs[i]);
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

	MatrixGenerator* matrix_generator = new MatrixGenerator();
	ITripleArgFunction* my_function = new MyFunction();
	int nr_of_tiers = 2;
	std::list<Tier*>* tier_list = matrix_generator->CreateMatrixAndRhs(nr_of_tiers, 0, 0, 0, 2, my_function);
					  
	double** matrix = matrix_generator->GetMatrix();
	double* rhs = matrix_generator->GetRhs();
	int size = matrix_generator->GetMatrixSize();

	//printf("%d\n\n",size);

	//print_matrix(matrix,size);



	/*
	EquationSystem* eq = new EquationSystem(matrix,rhs,size);
	eq->eliminate(size);
	eq->backwardSubstitute(size-1);
	print_rhs(eq->rhs,size);
	*/

	//print_matrix_rhs(matrix,rhs,size);

	printf("----------------------------------------------------------\n");



	// przyklad stworzenia i wypisywania macierzy frontalnych
	double** matrix2 = new double*[size];
	double* rhs2 = new double[size]();
	for(int i = 0; i<size; i++)
		matrix2[i] = new double[size]();
	
	std::list<Tier*>::iterator it = tier_list->begin();
	while(it != tier_list->end()){
		(*it)->FillMatrixAndRhs(matrix2,rhs2,size);
		++it;
	}
	//print_matrix_rhs(matrix2,rhs2,size);
	/*
	EquationSystem* eq2 = new EquationSystem(matrix2,rhs2,size);
	eq2->eliminate(size);
	eq2->backwardSubstitute(size-1);
	print_rhs(eq2->rhs,size);
	*/
	//GraphGenerator* generator = new GraphGenerator();
	//generator->GenerateGraph(100);

	/*
	NArgFunction* f3a = new VertexTopRightFarShapeFunction(true,0,0,0,1,1,1,BOT_LEFT_NEAR);

	std::vector<double> v;
	v.push_back(0.5);
	v.push_back(0.4);
	v.push_back(0.1);
	printf("wynik %lf\n",f3a->ComputeValue(v));
	printf("wynik %lf\n",f3a->ComputeValue(v));
	*/
	return 0;

}

