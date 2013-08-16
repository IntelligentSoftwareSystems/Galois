#include "Element.hxx"
#include <stdio.h>
using namespace D2;
Element** Element::CreateAnotherTier(int nr)
{

	double bot_element_xl = xl; 
	double bot_element_xr = xr;
	double bot_element_yl = yl - (yr-yl); 
	double bot_element_yr = yl;
		
	double right_element_xl = xr; 
	double right_element_xr = xr + (xr-xl); 	
	double right_element_yl = yl; 
	double right_element_yr = yr; 

	Element* right_element = new Element(right_element_xl,right_element_yl,right_element_xr,right_element_yr,TOP_RIGHT,false); 	
	Element* bot_element = new Element(bot_element_xl,bot_element_yl,bot_element_xr,bot_element_yr,BOT_LEFT,false); 	


	top_left_vertex_nr = nr + 2; 
	bot_left_vertex_nr = nr; 
	top_right_vertex_nr = nr + 4; 
	bot_right_vertex_nr = nr + 14; 
		
	left_edge_nr = nr + 1; 
	top_edge_nr = nr + 3; 
	right_edge_nr = nr + 9; 
	bot_edge_nr = nr + 7;
		
	interior_nr = nr + 8; 
		
	bot_element->set_bot_left_vertex_nr(nr); 
	bot_element->set_top_left_vertex_nr(nr + 2); 
	bot_element->set_bot_right_vertex_nr(nr + 12); 
	bot_element->set_top_right_vertex_nr(nr + 14); 
		
	bot_element->set_left_edge_nr(nr + 1); 
	bot_element->set_top_edge_nr(nr + 7); 
	bot_element->set_right_edge_nr(nr + 13); 
	bot_element->set_bot_edge_nr(nr + 5); 
		
	bot_element->set_interior_nr(nr + 6); 
		
	right_element->set_top_left_vertex_nr(nr + 2); 
	right_element->set_top_right_vertex_nr(nr + 4);
	right_element->set_bot_left_vertex_nr(nr + 14); 
	right_element->set_bot_right_vertex_nr(nr + 16); 
		
	right_element->set_left_edge_nr(nr + 9); 
	right_element->set_top_edge_nr(nr + 3); 
	right_element->set_right_edge_nr(nr + 11); 
	right_element->set_bot_edge_nr(nr + 15); 
		
	right_element->set_interior_nr(nr + 10); 

		
	Element** elements = new Element*[2]; 
	elements[0] = right_element;
	elements[1] = bot_element; 
	return elements;

}


Element** Element::CreateFirstTier(int nr){
		
	double bot_element_xl = xl; 
	double bot_element_xr = xr;
	double bot_element_yl = yl - (yr-yl); 
	double bot_element_yr = yl;
		
	double right_element_xl = xr; 
	double right_element_xr = xr + (xr-xl); 	
	double right_element_yl = yl; 
	double right_element_yr = yr; 

	Element* right_element = new Element(right_element_xl,right_element_yl,right_element_xr,right_element_yr,TOP_RIGHT,true); 	
	Element* bot_element = new Element(bot_element_xl,bot_element_yl,bot_element_xr,bot_element_yr,BOT_LEFT,true); 	
		
	top_left_vertex_nr = nr + 4; 
	bot_left_vertex_nr = nr + 2; 
	top_right_vertex_nr = nr + 6; 
	bot_right_vertex_nr = nr + 18; 
		
	left_edge_nr = nr + 3; 
	top_edge_nr = nr + 5; 
	right_edge_nr = nr + 13; 
	bot_edge_nr = nr + 11; 
	
	interior_nr = nr + 12; 
		
	bot_element->set_bot_left_vertex_nr(nr); 
	bot_element->set_top_left_vertex_nr(nr + 2); 
	bot_element->set_bot_right_vertex_nr(nr + 16); 
	bot_element->set_top_right_vertex_nr(nr + 18); 
	
	bot_element->set_left_edge_nr(nr + 1); 
	bot_element->set_top_edge_nr(nr + 11); 
	bot_element->set_right_edge_nr(nr + 17); 
	bot_element->set_bot_edge_nr(nr + 9); 
	
	bot_element->set_interior_nr(nr + 10); 	
		
	right_element->set_top_left_vertex_nr(nr + 6); 
	right_element->set_top_right_vertex_nr(nr + 8); 
	right_element->set_bot_left_vertex_nr(nr + 18); 
	right_element->set_bot_right_vertex_nr(nr + 20); 
	
	right_element->set_left_edge_nr(nr + 13); 
	right_element->set_top_edge_nr(nr + 7); 
	right_element->set_right_edge_nr(nr + 15); 
	right_element->set_bot_edge_nr(nr + 19); 
	
	right_element->set_interior_nr(nr + 14); 
		
		
	Element** elements = new Element*[2]; 
	elements[0] = right_element; 
	elements[1] = bot_element; 	

	return elements;
		
}


Element** Element::CreateLastTier(int nr)
{	
		top_left_vertex_nr = nr + 2; 
		bot_left_vertex_nr = nr; 
		top_right_vertex_nr = nr + 4; 
		bot_right_vertex_nr = nr + 8; 
		
		left_edge_nr = nr + 1; 
		top_edge_nr = nr + 3; 
		right_edge_nr = nr + 7; 
		bot_edge_nr = nr + 5; 

		interior_nr = nr + 6; 

		return new Element*[0];
		
}

class DoubleArgFunctionProduct : public IDoubleArgFunction
{
	
	private:
		IDoubleArgFunction* function1; 
		IDoubleArgFunction* function2; 
		
	public:
		void SetFunctions(IDoubleArgFunction* _function1, IDoubleArgFunction* _function2)
		{
			function1 = _function1;
			function2 = _function2;
		}
	
		virtual double ComputeValue(double x, double y)
		{
			return function1->ComputeValue(x,y)*function2->ComputeValue(x,y); 
		}
	
};


void Element::comp(int indx1, int indx2, IDoubleArgFunction* f1, IDoubleArgFunction* f2,double** matrix)
{
		
		DoubleArgFunctionProduct* product = new DoubleArgFunctionProduct();
		product->SetFunctions(f1,f2);
		matrix[indx1][indx2] += GaussianQuadrature::definiteDoubleIntegral(xl, xr, yl, yr, product);
		delete product;

}


void Element::fillMatrix(double** matrix)
{
	fillMatrix(matrix,0);
		
}

void Element::fillMatrix(double** matrix, int start_adj_nr)
{
	int functionNumbers[] = { bot_left_vertex_nr, left_edge_nr, top_left_vertex_nr, top_edge_nr, top_right_vertex_nr, bot_edge_nr, interior_nr, right_edge_nr, bot_right_vertex_nr };
					
		for(int i = 0; i<9; i++){
			for(int j = 0; j<9; j++){
				comp(functionNumbers[i] - start_adj_nr, functionNumbers[j] - start_adj_nr,
						shapeFunctions[i], shapeFunctions[j], matrix);
			}
		}
}

void Element::fillRhs(double* rhs, IDoubleArgFunction* f){
	fillRhs(rhs, f, 0);
}

void Element::fillRhs(double* rhs, IDoubleArgFunction* f, int start_adj_nr)
{
		int functionNumbers[] = { bot_left_vertex_nr, left_edge_nr, top_left_vertex_nr, top_edge_nr, top_right_vertex_nr, bot_edge_nr, interior_nr, right_edge_nr, bot_right_vertex_nr }; 
		for(int i = 0; i<9; i++){
			DoubleArgFunctionProduct* product = new DoubleArgFunctionProduct();
			product->SetFunctions(shapeFunctions[i], f);
			
			rhs[functionNumbers[i] - start_adj_nr] += GaussianQuadrature::definiteDoubleIntegral(xl, xr, yl, yr, product);
			delete product; 
		}		
}

void Element::fillTierMatrix(double** matrix, double* rhs, IDoubleArgFunction* f, int start_nr_adj){
		fillMatrix(matrix, start_nr_adj);
		fillRhs(rhs, f, start_nr_adj);
}

bool Element::checkSolution(std::map<int,double> *solution_map, IDoubleArgFunction* f)
{
	int nr_of_nodes = 9;
	double coefficients[nr_of_nodes];


	coefficients[0] = solution_map->find(bot_left_vertex_nr)->second;
	coefficients[1] = solution_map->find(left_edge_nr)->second;
	coefficients[2] = solution_map->find(top_left_vertex_nr)->second;
	coefficients[3] = solution_map->find(top_edge_nr)->second;
	coefficients[4] = solution_map->find(top_right_vertex_nr)->second;
	coefficients[5] = solution_map->find(bot_edge_nr)->second;
	coefficients[6] = solution_map->find(interior_nr)->second;
	coefficients[7] = solution_map->find(right_edge_nr)->second;
	coefficients[8] = solution_map->find(bot_right_vertex_nr)->second;

	int nr_of_samples = 5;
	double epsilon = 1e-8;

	double rnd_x_within_element;
	double rnd_y_within_element;

	for(int i = 0; i<nr_of_samples; i++)
	{
		double value = 0;
		double rnd_x_within_element = ((double) rand() / (RAND_MAX))*(xr-xl) + xl;
		double rnd_y_within_element = ((double) rand() / (RAND_MAX))*(yr-yl) + yl;
		for(int i = 0; i<nr_of_nodes; i++)
			value+=coefficients[i]*shapeFunctions[i]->ComputeValue(rnd_x_within_element,rnd_y_within_element);
		printf("%d Checking at: %lf %lf values: %lf %lf\n",position,rnd_x_within_element,rnd_y_within_element,value,f->ComputeValue(rnd_x_within_element,rnd_y_within_element));
		if(fabs(value - f->ComputeValue(rnd_x_within_element,rnd_y_within_element)) > epsilon)
		{
			for(int i = 0; i<9; i++)
				printf("%lf %d\n",coefficients[i],bot_right_vertex_nr);
			return false;
		}
	}

	return true;
}
