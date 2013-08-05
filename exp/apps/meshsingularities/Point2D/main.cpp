#include <stdio.h>
#include "MatrixGenerator.hxx"

class MyFunction : public IDoubleArgFunction
{
	public:
		virtual double ComputeValue(double x, double y)
		{
			return x*y + (1-x)*y + x*x+y*y; 
		}
};

void print_matrix_rhs(double** matrix, double* rhs, int size)
{
	for(int i = 0; i<size; i++){
		for(int j = 0; j<size; j++){
				printf("%0.3lf ",matrix[i][j]);
		}
		printf("| %0.3lf\n",rhs[i]);

	}
}

int main(int argc, char** argv)
{

	MatrixGenerator* matrix_generator = new MatrixGenerator();
	IDoubleArgFunction* my_function = new MyFunction();
	
	std::list<Tier*>* tier_list = matrix_generator->CreateMatrixAndRhs(2, -1, -1, 4, my_function);
					  
	double** matrix = matrix_generator->GetMatrix();
	double* rhs = matrix_generator->GetRhs(); 
	int size = matrix_generator->GetMatrixSize(); 
	printf("%d\n\n",size); 
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
	print_matrix_rhs(matrix2,rhs2,size);
	return 0; 
}

