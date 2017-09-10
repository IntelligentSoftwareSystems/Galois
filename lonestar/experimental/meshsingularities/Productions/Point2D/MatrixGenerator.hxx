#ifndef __MATRIXGENERATOR_2D_H_INCLUDED
#define __MATRIXGENERATOR_2D_H_INCLUDED

#include "DoubleArgFunction.hxx"
#include "Element.hxx"
#include "EPosition.hxx"
#include "Tier.hxx"
#include "../MatrixGeneration/GenericMatrixGenerator.hxx"
#include <vector>
#include <map>
namespace D2{
class MatrixGenerator : public GenericMatrixGenerator{
		
	private:
		std::list<Element*> element_list;


	public:
		virtual std::vector<EquationSystem*>* CreateMatrixAndRhs(TaskDescription& task_description);
		virtual void checkSolution(std::map<int,double> *solution_map, double (*f)(int dim, ...));

		virtual std::vector<int>* GetProductionParameters(int polynomial_degree)
		{
			std::vector<int>* vec = new std::vector<int>(4);
			(*vec)[0] = 5;
			(*vec)[1] = 17;
			(*vec)[2] = 21;
			(*vec)[3] = 21;
			return vec;
		}

		virtual ~MatrixGenerator(){
			std::list<Element*>::iterator it_e = element_list.begin();
			for(; it_e != element_list.end(); ++it_e){
				delete *it_e;
			}

		}
};
}
#endif
