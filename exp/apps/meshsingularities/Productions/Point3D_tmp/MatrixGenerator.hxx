#ifndef __MATRIXGENERATOR_H_INCLUDED
#define __MATRIXGENERATOR_H_INCLUDED

#include "TripleArgFunction.hxx"
#include "Element.hxx"
#include "EPosition.hxx"
#include "Tier.hxx"
#include <list>
namespace tmp{
class MatrixGenerator {
		
	private:
		double** matrix; 
		double* rhs; 
		int matrix_size;
		std::list<Element*> element_list;
		std::list<Tier*>* tier_list;

	public:
		std::list<Tier*>* CreateMatrixAndRhs(int nr_of_tiers, double bot_left_x,
				double bot_left_y, double bot_left_z, double size, ITripleArgFunction* f);
		
		double** GetMatrix()
		{
			return matrix;
		}

		double* GetRhs()
		{
			return rhs;
		}
		
		int GetMatrixSize()
		{
			return matrix_size;
		}

		~MatrixGenerator(){
			for(int i = 0; i<matrix_size; i++)
				delete[] matrix[i];
			delete[] matrix;
			delete[] rhs;

			std::list<Element*>::iterator it_e = element_list.begin();
			for(; it_e != element_list.end(); ++it_e){
				delete *it_e;
			}

			std::list<Tier*>::iterator it_t = tier_list->begin();
			for(; it_t != tier_list->end(); ++it_t)
				delete *it_t;

			delete tier_list;
		}
};
}
#endif
