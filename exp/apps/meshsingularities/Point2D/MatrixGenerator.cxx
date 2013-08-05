#include "MatrixGenerator.hxx"
#include <stdio.h>



std::list<Tier*>* MatrixGenerator::CreateMatrixAndRhs(int nr_of_tiers, double bot_left_x, double bot_left_y, double size, IDoubleArgFunction* f)
{	
		
		int nr = 0; 
		
		Element* element = new Element(bot_left_x, bot_left_y + size / 2.0, bot_left_x + size / 2.0, bot_left_y + size, TOP_LEFT, true);
		
		Element** elements = element->CreateFirstTier(nr);
		
		element_list.push_back(element);
		element_list.push_back(elements[0]);
		element_list.push_back(elements[1]);
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
			
			element_list.push_back(element);
			element_list.push_back(elements[0]);
			element_list.push_back(elements[1]);
			nr+=12;
			
		}
		
		x = x + s; 
		y = y - s;
		element = new Element(x, y, x + s, y + s, BOT_RIGHT, false);
		element->CreateLastTier(nr);
		element_list.push_back(element);
		
		matrix_size = 9 + nr_of_tiers*12 + 4; 
		rhs = new double[matrix_size];
		matrix = new double*[matrix_size];
		for(int i = 0; i<matrix_size; i++)
			matrix[i] = new double[matrix_size];
		
		std::list<Element*>::iterator it = element_list.begin();
		while(it != element_list.end())
		{
				Element* matrix_creation_element = (*it);
				matrix_creation_element->fillMatrix(matrix);
				matrix_creation_element->fillRhs(rhs, f);
				++it;
		}

		std::list<Tier*>* tier_list = new std::list<Tier*>();
		it = element_list.begin();

		for(int i = 0; i<nr_of_tiers; i++){
			Tier* tier;
			if(i == nr_of_tiers -1){
				tier = new Tier(*it,*(++it),*(++it),*(++it),f);
			}
			else{
				tier = new Tier(*it,*(++it),*(++it),NULL,f);
				++it;
			}
			tier_list->push_back(tier);
		}

		return tier_list;
}

