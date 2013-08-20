/*
 * Processing.h
 *
 *  Created on: Aug 20, 2013
 *      Author: kjopek
 */

#ifndef PROCESSING_H_
#define PROCESSING_H_

#include "Vertex.h"
#include "EquationSystem.h"
#include "Production.h"

#include <vector>
#include <list>

class Processing {
public:
	Processing();

	/* preprocess the input and convert data to form needed by production,
	   this is sequential code */
	std::vector<EquationSystem *>* preprocess(std::list<EquationSystem*>* input,
			AbstractProduction *productions);

	/* pull data from leafs and return the solution of MES
	   this is sequential code*/
	std::vector<double> *postprocess(std::vector<Vertex *> *leafs,
			std::vector<EquationSystem *> *inputData,
			AbstractProduction *productions);

	virtual ~Processing();

	/* these methods may be useful in order to parallelize {pre,post}processing */
	EquationSystem *preprocessA1(EquationSystem *input, AbstractProduction *productions);
	EquationSystem *preprocessA(EquationSystem *input, AbstractProduction *productions);
	EquationSystem *preprocessAN(EquationSystem *input, AbstractProduction *productions);

	void postprocessA1(Vertex *leaf, EquationSystem *inputData,
			AbstractProduction *productions, std::vector<double> *result, int num);
	void postprocessA(Vertex *leaf, EquationSystem *inputData,
			AbstractProduction *productions, std::vector<double> *result, int num);
	void postprocessAN(Vertex *leaf, EquationSystem *inputData,
			AbstractProduction *productions, std::vector<double> *result, int num);
};

#endif /* PROCESSING_H_ */
