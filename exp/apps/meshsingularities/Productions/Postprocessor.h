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

class Postprocessor2D : public Postprocessor {
// everything same
};

class Postprocessor3D : public Postprocessor {
public:
	std::vector<double> *postprocess(std::vector<Vertex *> *leafs,
									 std::vector<EquationSystem *> *inputData,
									 AbstractProduction *productions);
};

#endif /* POSTPROCESSOR_H_ */
