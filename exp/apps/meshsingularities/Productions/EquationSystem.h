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
  private:
        // needed because of implementation of swapRows
        double *origPtr;
  public:
	// this variables _should_ be public
	// Productions will use them directly

	int n;
	double ** matrix;
	double *rhs;

	EquationSystem() {} ;
	EquationSystem(int n);
	EquationSystem(double ** matrix, double *rhs, int size);
	virtual ~EquationSystem();

	void swapRows(const int i, const int j);
	void swapCols(const int i, const int j);

	void eliminate(const int rows);
	void backwardSubstitute(const int startingRow);

	void print() const;
};

#endif /* EQUATIONSYSTEM_H_ */
