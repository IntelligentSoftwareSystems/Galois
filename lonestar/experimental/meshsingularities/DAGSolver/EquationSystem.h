/*
 * EquationSystem.h
 *
 * Implementation of dense matrix storage with column-major order
 *
 *  Created on: 05-08-2013
 *      Author: kj
 */

#ifndef EQUATIONSYSTEM_H_
#define EQUATIONSYSTEM_H_

#include <cstdio>
#include <cstdlib>

enum SolverMode { OLD, LU, CHOLESKY };

class EquationSystem {
private:
  // needed because of implementation of swapRows
  double* origPtr;
  SolverMode mode = OLD;

public:
  // this variables _should_ be public
  // Productions will use them directly

  unsigned long n;
  double** matrix;
  double* rhs;

  EquationSystem(){};
  EquationSystem(unsigned long n, SolverMode mode = OLD);
  virtual ~EquationSystem();

  void swapRows(const int i, const int j);
  void swapCols(const int i, const int j);

  int eliminate(const int rows);
  void backwardSubstitute(const int startingRow);

  void checkRow(int row_nr, int* values, int values_cnt);
  void print() const;
};

#endif /* EQUATIONSYSTEM_H_ */
