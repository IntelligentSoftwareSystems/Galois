#include "MatrixGenerator.hxx"


using namespace D3;
std::list<Tier*>* MatrixGenerator::CreateMatrixAndRhs(int nr_of_tiers, double bot_left_near_x,
		double bot_left_near_y, double bot_left_near_z, double size, ITripleArgFunction* f)
{	
		
		int nr = 0; 
		
		Element* element = new Element(bot_left_near_x, bot_left_near_y + size / 2.0, bot_left_near_z,
				 size / 2.0, BOT_LEFT_FAR, true);
		
		Element** elements = element->CreateFirstTier(nr);
		
		for(int i = 0; i<7; i++)
			element_list.push_back(elements[i]);

		delete [] elements;
		//nr of degrees of freedom in first tier which are not in 2
		nr+=98;
		
		double x = bot_left_near_x;
		double y = bot_left_near_y + size / 2.0;
		double z = bot_left_near_z;
		double s = size / 2.0;
		
		for(int i = 1; i < nr_of_tiers ; i++){
			
			x = x + s; 
			y = y - s / 2.0;
			s = s /2.0;
			element = new Element(x, y, z, s, BOT_LEFT_FAR, false);
			elements = element->CreateAnotherTier(nr);
			
			for(int i = 0; i<7; i++)
				element_list.push_back(elements[i]);
			//nr of degrees of freedom in i tier which are not in i + 1
			delete [] elements;
			nr+=56;
			
		}
		
		x = x + s; 
		y = y - s;
		element = new Element(x, y, z, s, BOT_RIGHT_NEAR, false);
		element->CreateLastTier(nr);
		element_list.push_back(element);
		
		matrix_size = 98 + 56*(nr_of_tiers-1) + 27;
		rhs = new double[matrix_size]();
		matrix = new double*[matrix_size];
		for(int i = 0; i<matrix_size; i++)
			matrix[i] = new double[matrix_size]();



		tier_list = new std::list<Tier*>();
		std::list<Element*>::iterator it = element_list.begin();
		it = element_list.begin();

		for(int i = 0; i<nr_of_tiers; i++){
			Tier* tier;
			if(i == nr_of_tiers -1){
				tier = new Tier(*it,*(++it),*(++it),*(++it),*(++it),*(++it),*(++it),*(++it),f,matrix,rhs);
			}
			else{
				tier = new Tier(*it,*(++it),*(++it),*(++it),*(++it),*(++it),*(++it),NULL,f,matrix,rhs);
				++it;
			}
			tier_list->push_back(tier);
		}

		return tier_list;
}

void MatrixGenerator::checkSolution(std::map<int,double> *solution_map, ITripleArgFunction* f)
{
	std::list<Element*>::iterator it = element_list.begin();
	bool solution_ok = true;
	while(it != element_list.end() && solution_ok)
	{
		Element* element = (*it);
		solution_ok = element->checkSolution(solution_map,f);
		++it;
	}
	if(solution_ok)
		printf("SOLUTION OK\n");
	else
		printf("WRONG SOLUTION\n");

}

