/*
 * Functions.h
 *
 *  Created on: Aug 22, 2013
 *      Author: kjopek
 */

#ifndef FUNCTIONS_H_
#define FUNCTIONS_H_

#include <functional>

double f1(int, ...);
double f2(int, ...);
double f3(int, ...);
double f4(int, ...);

enum Functions { F1, F2, F3, F4 };

struct Function {
  const char* name;
  double (*func)(int, ...);
};

extern const Function functionsTable[];

#endif /* FUNCTIONS_H_ */
