/*
 * MatrixGenerator.hxx
 *
 *  Created on: Aug 22, 2013
 *      Author: dgoik
 */

#ifndef MATRIXGENERATOR_HXX_
#define MATRIXGENERATOR_HXX_
#include <cstdarg>
#include <boost/function.hpp>
#include <boost/bind.hpp>
#include <list>
#include <map>
#include <stdio.h>
#include "../EquationSystem.h"
#include "../GaloisWorker.h"

class GenericMatrixGenerator {

	protected:
		double** matrix;
		double* rhs;
		int matrix_size;
		std::list<EquationSystem*>* tier_list;

	public:
		virtual std::list<EquationSystem*>* CreateMatrixAndRhs(TaskDescription& task_description) = 0;
		virtual void checkSolution(std::map<int,double> *solution_map, double (*f)(int dim, ...)) = 0;

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

		virtual std::vector<int>* GetProductionParameters(int polynomial_degree)
		{
			return new std::vector<int>();
		}

		virtual ~GenericMatrixGenerator(){
			for(int i = 0; i<matrix_size; i++)
				delete[] matrix[i];
			delete[] matrix;
			delete[] rhs;

			std::list<EquationSystem*>::iterator it_t = tier_list->begin();
			for(; it_t != tier_list->end(); ++it_t)
				delete *it_t;

			delete tier_list;
		}

};


#endif /* MATRIXGENERATOR_HXX_ */
