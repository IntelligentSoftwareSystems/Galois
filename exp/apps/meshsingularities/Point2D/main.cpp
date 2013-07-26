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

int main(int argc, char** argv)
{
	printf("ok\n"); 
	MatrixGenerator* matrix_generator = new MatrixGenerator();
	printf("ok\n"); 
	IDoubleArgFunction* my_function = new MyFunction();
	
	matrix_generator->CreateMatrixAndRhs(3, -1, -1, 4, my_function);
					  
	double** matrix = matrix_generator->GetMatrix();
	double* rhs = matrix_generator->GetRhs(); 
	int size = matrix_generator->GetMatrixSize(); 
	printf("%d\n\n",size); 
	
	for(int i = 0; i<size; i++){
		for(int j = 0; j<size; j++){
				printf("%0.3lf ",matrix[i][j]); 
		}
		printf("| %0.3lf\n",rhs[i]); 
		
	}
	printf("ok2\n");	
	
	return 0; 
}

