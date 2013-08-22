#ifndef __MATRIXGENERATOR_2D_H_INCLUDED
#define __MATRIXGENERATOR_2D_H_INCLUDED

#include "DoubleArgFunction.hxx"
#include "Element.hxx"
#include "EPosition.hxx"
#include "Tier.hxx"
#include "../MatrixGeneration/GenericMatrixGenerator.hxx"
#include <list>
#include <map>
namespace D2{
class MatrixGenerator : public GenericMatrixGenerator{
		
	private:
		std::list<Element*> element_list;


	public:
		virtual std::list<EquationSystem*>* CreateMatrixAndRhs(TaskDescription& task_description);
		virtual void checkSolution(std::map<int,double> *solution_map, double (*f)(int dim, ...));


		virtual ~MatrixGenerator(){
			std::list<Element*>::iterator it_e = element_list.begin();
			for(; it_e != element_list.end(); ++it_e){
				delete *it_e;
			}

		}
};
}
#endif
