/*
 * NumaEquationSystem.h
 *
 *  Created on: Aug 30, 2013
 *      Author: kjopek
 */

#ifndef NUMAEQUATIONSYSTEM_H_
#define NUMAEQUATIONSYSTEM_H_

#include <cstdio>
#include <cstdlib>

#include "EquationSystem.h"

class NumaEquationSystem : public EquationSystem {
public:
  // this variables _should_ be public
  // Productions will use them directly

  NumaEquationSystem(int n, int node);
  NumaEquationSystem(double** matrix, double* rhs, int size, int node);
  virtual ~NumaEquationSystem();
};

#endif /* NUMAEQUATIONSYSTEM_H_ */
