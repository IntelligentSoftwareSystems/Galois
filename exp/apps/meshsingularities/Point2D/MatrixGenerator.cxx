#include "MatrixGenerator.hxx"
#include <stdio.h>



void MatrixGenerator::CreateMatrixAndRhs(int nr_of_tiers, double bot_left_x, double bot_left_y, double size, IDoubleArgFunction* f)
{	
		
		int nr = 0; 
		std::stack<Element*> element_stack;
		
		
		Element* element = new Element(bot_left_x, bot_left_y + size / 2.0, bot_left_x + size / 2.0, bot_left_y + size, TOP_LEFT, true);
		
		Element** elements = element->CreateFirstTier(nr);
		
		element_stack.push(element);
		element_stack.push(elements[0]);
		element_stack.push(elements[1]);
		nr+=16; 
		
		double x = bot_left_x;
		double y = bot_left_y + size / 2.0;
		double s = size / 2.0;
		
		for(int i = 1; i < nr_of_tiers ; i++){
			
			x = x + s; 
			y = y - s / 2.0;
			s = s /2.0;
			element = new Element(x,y, x + s, y + s, TOP_LEFT, false);
			elements = element->CreateAnotherTier(nr);
			
			element_stack.push(element);
			element_stack.push(elements[0]);
			element_stack.push(elements[1]);
			nr+=12;
			
		}
		
		x = x + s; 
		y = y - s;
		element = new Element(x, y, x + s, y + s, BOT_RIGHT, false);
		element->CreateLastTier(nr);
		element_stack.push(element);
		
		matrix_size = 9 + nr_of_tiers*12 + 4; 
		rhs = new double[matrix_size];
		matrix = new double*[matrix_size];
		for(int i = 0; i<matrix_size; i++)
			matrix[i] = new double[matrix_size];
		

		while(!element_stack.empty())
		{
			printf("ITERACJA \n\n"); 
			
			Element* matrix_creation_element = element_stack.top();
			matrix_creation_element->fillMatrix(matrix); 
			matrix_creation_element->fillRhs(rhs, f); 
			element_stack.pop();
			

		}
	
}

class MyFunction : public IDoubleArgFunction
{
	public:
		virtual double ComputeValue(double x, double y)
		{
			return 2; 
		}
};
