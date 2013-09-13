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
#include "PointProduction.hxx"

#include <vector>
#include <list>

class Processing {
public:
	Processing();

	/* preprocess the input and convert data to form needed by production,
	   this is sequential code */
	std::vector<EquationSystem *>* preprocess(std::list<EquationSystem*>* input,
			PointProduction *productions) const;

	/* pull data from leafs and return the solution of MES
	   this is sequential code*/
	std::vector<double> *postprocess(std::vector<Vertex *> *leafs,
			std::vector<EquationSystem *> *inputData,
			PointProduction *productions) const;

	virtual ~Processing();

	/* these methods may be useful in order to parallelize {pre,post}processing */
	EquationSystem *preprocessA1(EquationSystem *input, PointProduction *productions) const;
	EquationSystem *preprocessA(EquationSystem *input, PointProduction *productions) const;
	EquationSystem *preprocessAN(EquationSystem *input, PointProduction *productions) const;

	void postprocessA1(Vertex *leaf, EquationSystem *inputData,
			PointProduction *productions, std::vector<double> *result, int num) const;
	void postprocessA(Vertex *leaf, EquationSystem *inputData,
			PointProduction *productions, std::vector<double> *result, int num) const;
	void postprocessAN(Vertex *leaf, EquationSystem *inputData,
			PointProduction *productions, std::vector<double> *result, int num) const;

	// iterate over leafs and return them in correct order
	std::vector<Vertex*> *collectLeafs(Vertex *p);
};

#endif /* PROCESSING_H_ */
