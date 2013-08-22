#include "MatrixGenerator.hxx"

using namespace D2;
std::list<EquationSystem*>* MatrixGenerator::CreateMatrixAndRhs(TaskDescription& task_description)
{	

		IDoubleArgFunction* f = new DoubleArgFunctionWrapper(task_description.function);
		double bot_left_x = task_description.x;
		double bot_left_y = task_description.y;
		int nr_of_tiers = task_description.nrOfTiers;
		double size = task_description.size;

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
		rhs = new double[matrix_size]();
		matrix = new double*[matrix_size];
		for(int i = 0; i<matrix_size; i++)
			matrix[i] = new double[matrix_size]();

		tier_list = new std::list<EquationSystem*>();
		std::list<Element*>::iterator it = element_list.begin();
		it = element_list.begin();

		for(int i = 0; i<nr_of_tiers; i++){
			Tier* tier;
			if(i == nr_of_tiers -1){
				tier = new Tier(*it,*(++it),*(++it),*(++it),f,matrix,rhs);
			}
			else{
				tier = new Tier(*it,*(++it),*(++it),NULL,f,matrix,rhs);
				++it;
			}
			tier_list->push_back(tier);
		}
		return tier_list;
}

void MatrixGenerator::checkSolution(std::map<int,double> *solution_map, double (*function)(int dim, ...))
{
	IDoubleArgFunction* f = new DoubleArgFunctionWrapper(*function);
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

