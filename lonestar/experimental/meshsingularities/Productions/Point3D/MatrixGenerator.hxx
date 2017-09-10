#ifndef __MATRIXGENERATOR_3D_H_INCLUDED
#define __MATRIXGENERATOR_3D_H_INCLUDED

#include "TripleArgFunction.hxx"
#include "Element.hxx"
#include "EPosition.hxx"
#include "Tier.hxx"
#include "../MatrixGeneration/GenericMatrixGenerator.hxx"
#include <map>
#include <vector>
namespace D3{
class MatrixGenerator : public GenericMatrixGenerator {
		
	private:
		std::list<Element*> element_list;

	public:
		virtual std::vector<EquationSystem*>* CreateMatrixAndRhs(TaskDescription& task_description);
		virtual void checkSolution(std::map<int,double> *solution_map, double (*f)(int dim, ...));

		virtual std::vector<int>* GetProductionParameters(int polynomial_degree)
		{
			std::vector<int>* vec = new std::vector<int>(4);
			(*vec)[0] = 19;
			(*vec)[1] = 75;
			(*vec)[2] = 117;
			(*vec)[3] = 83;
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
