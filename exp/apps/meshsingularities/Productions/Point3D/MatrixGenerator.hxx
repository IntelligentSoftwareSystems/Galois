#ifndef __MATRIXGENERATOR_3D_H_INCLUDED
#define __MATRIXGENERATOR_3D_H_INCLUDED

#include "TripleArgFunction.hxx"
#include "Element.hxx"
#include "EPosition.hxx"
#include "Tier.hxx"
#include "../MatrixGeneration/GenericMatrixGenerator.hxx"
#include <list>
#include <map>
namespace D3{
class MatrixGenerator : public GenericMatrixGenerator {
		
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
