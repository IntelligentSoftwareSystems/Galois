/*
 * EquationSystem.h
 *
 *  Created on: 05-08-2013
 *      Author: kj
 */

#ifndef EQUATIONSYSTEM_H_
#define EQUATIONSYSTEM_H_

#include <cstdio>
#include <cstdlib>

class EquationSystem {
  public:
	// this variables _should_ be public
	// Productions will use them directly

	int n;
	double ** matrix;
	double *rhs;

	EquationSystem(int n);
	EquationSystem(double ** matrix, double *rhs, int size);
	virtual ~EquationSystem();

	void swapRows(int i, int j);
	void swapCols(int i, int j);

	void eliminate(int rows);
	void backwardSubstitute(int startingRow);

	void print();
};

#endif /* EQUATIONSYSTEM_H_ */
