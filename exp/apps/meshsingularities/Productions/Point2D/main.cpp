#include <stdio.h>
#include "MatrixGenerator.hxx"
#include <map>
#include <time.h>

using namespace D2;
class MyFunction : public IDoubleArgFunction
{
	public:
		virtual double ComputeValue(double x, double y)
		{
			return 1;
		}
};

void print_matrix_rhs(double** matrix, double* rhs, int size)
{
	for(int i = 0; i<size; i++){
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


int main(int argc, char** argv)
{

	MatrixGenerator* matrix_generator = new MatrixGenerator();
	IDoubleArgFunction* my_function = new MyFunction();
	
	std::list<Tier*>* tier_list = matrix_generator->CreateMatrixAndRhs(2, -1, -1, 4, my_function);
	srand(time(NULL));
	double** matrix = matrix_generator->GetMatrix();
	double* rhs = matrix_generator->GetRhs(); 
	int size = matrix_generator->GetMatrixSize(); 
	std::map<int,double>* mapp = new std::map<int,double>();
	(*mapp)[0] = 1.0;
	(*mapp)[1] = 0.0;
	(*mapp)[2] = 1.0;
	(*mapp)[3] = 0.0;
	(*mapp)[4] = 1.0;
	(*mapp)[5] = 0.0;
	(*mapp)[6] = 1.0;
	(*mapp)[7] = 0.0;
	(*mapp)[8] = 1.0;
	(*mapp)[9] = 0.0;
	(*mapp)[10] = 0.0;
	(*mapp)[11] = 0.0;
	(*mapp)[12] = 0.0;
	(*mapp)[13] = 0.0;
	(*mapp)[14] = 0.0;
	(*mapp)[15] = 0.0;
	(*mapp)[16] = 1.0;
	(*mapp)[17] = 0.0;
	(*mapp)[18] = 1.0;
	(*mapp)[19] = 0.0;
	(*mapp)[20] = 1.0;
	(*mapp)[28] = 1.0;
	(*mapp)[30] = 1.0;
	(*mapp)[32] = 1.0;
	(*mapp)[36] = 1.0;

	matrix_generator->checkSolution(mapp,my_function);

	//print_matrix(matrix,size);

	//print_rhs(rhs,size);
	//print_matrix_rhs(matrix,rhs,size);
	/*
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
	print_matrix_rhs(matrix2,rhs2,size);

	*/

	return 0;

}

