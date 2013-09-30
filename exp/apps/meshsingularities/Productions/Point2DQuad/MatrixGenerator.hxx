#ifndef __MATRIXGENERATOR_2DQUAD_H_INCLUDED
#define __MATRIXGENERATOR_2DQUAD_H_INCLUDED

#include "../Point2D/DoubleArgFunction.hxx"
#include "../Point2D/Element.hxx"
#include "../Point2D/EPosition.hxx"
#include "Tier.hxx"
#include "../MatrixGeneration/GenericMatrixGenerator.hxx"
#include <vector>
#include <map>
namespace D2Quad{
class MatrixGenerator : public GenericMatrixGenerator{
		
	private:
		std::list<Element*> element_list;
		Element** CreateElements(double x, double y, double element_size, int tier_size, bool first_tier);
		void CreateTier(double x, double y, double element_size, int tier_size, int function_nr, IDoubleArgFunction* f, bool first_tier);

	public:
		virtual std::vector<EquationSystem*>* CreateMatrixAndRhs(TaskDescription& task_description);
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
