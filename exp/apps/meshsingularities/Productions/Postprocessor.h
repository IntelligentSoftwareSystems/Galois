/*
 * Postprocessor.h
 *
 *  Created on: 14-08-2013
 *      Author: kj
 */

#ifndef POSTPROCESSOR_H_
#define POSTPROCESSOR_H_

#include <vector>
#include "Vertex.h"
#include "Production.h"

class Postprocessor {
public:
	virtual std::vector<double> *postprocess(std::vector<Vertex *> *leafs,
											 std::vector<EquationSystem *> *inputData,
											 AbstractProduction *productions);
	virtual ~Postprocessor();
};

#endif /* POSTPROCESSOR_H_ */
