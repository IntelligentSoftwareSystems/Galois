/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 */

#ifndef __DOUBLEARGFUNCTION_H_INCLUDED
#define __DOUBLEARGFUNCTION_H_INCLUDED
#include "EPosition.hxx"
#include "NPosition.hxx"
#include "../MatrixGeneration/IFunction.hxx"
#include <stdio.h>

namespace D2 {

double get_chi1(double var);
double get_chi2(double var);
double get_chi3(double var);

class IDoubleArgFunction : public IFunction {

public:
  virtual double ComputeValue(double x, double y) = 0;

  IDoubleArgFunction(double* coordinates, bool* neighbours)
      : IFunction(coordinates, neighbours) {}
  IDoubleArgFunction() {}
  virtual ~IDoubleArgFunction() {}
};

class DoubleArgFunctionWrapper : public IDoubleArgFunction {
private:
  double (*f)(int, ...);

public:
  virtual double ComputeValue(double x, double y) { return (*f)(2, x, y); }

  DoubleArgFunctionWrapper(double (*f)(int, ...)) : f(f) {}

  virtual ~DoubleArgFunctionWrapper() {}
};

class ShapeFunction : public IDoubleArgFunction {
protected:
  double xl;
  double yl;
  double xr;
  double yr;
  EPosition position;

  double getXValueOnElement(double x) { return (x - xl) / (xr - xl); }

  double getYValueOnElement(double y) { return (y - yl) / (yr - yl); }

public:
  ShapeFunction(double* coordinates, bool* neighbours, EPosition position)
      : IDoubleArgFunction(coordinates, neighbours), position(position) {
    xl = coordinates[0];
    xr = coordinates[1];
    yl = coordinates[2];
    yr = coordinates[3];
  }
};

class VertexBotLeftShapeFunction : public ShapeFunction {
public:
  VertexBotLeftShapeFunction(double* coordinates, bool* neighbours,
                             EPosition position)
      : ShapeFunction(coordinates, neighbours, position) {}

  virtual double ComputeValue(double x, double y);

  ~VertexBotLeftShapeFunction() {}
};

class VertexTopLeftShapeFunction : public ShapeFunction {
public:
  VertexTopLeftShapeFunction(double* coordinates, bool* neighbours,
                             EPosition position)
      : ShapeFunction(coordinates, neighbours, position) {}

  virtual double ComputeValue(double x, double y);
  ~VertexTopLeftShapeFunction() {}
};

class VertexTopRightShapeFunction : public ShapeFunction {
public:
  VertexTopRightShapeFunction(double* coordinates, bool* neighbours,
                              EPosition position)
      : ShapeFunction(coordinates, neighbours, position) {}

  virtual double ComputeValue(double x, double y);
  virtual ~VertexTopRightShapeFunction() {}
};

class VertexBotRightShapeFunction : public ShapeFunction {
public:
  VertexBotRightShapeFunction(double* coordinates, bool* neighbours,
                              EPosition position)
      : ShapeFunction(coordinates, neighbours, position) {}

  virtual double ComputeValue(double x, double y);
  ~VertexBotRightShapeFunction() {}
};

class EdgeLeftShapeFunction : public ShapeFunction {
public:
  EdgeLeftShapeFunction(double* coordinates, bool* neighbours,
                        EPosition position)
      : ShapeFunction(coordinates, neighbours, position) {}

  virtual double ComputeValue(double x, double y);
  ~EdgeLeftShapeFunction() {}
};

class EdgeTopShapeFunction : public ShapeFunction {
public:
  EdgeTopShapeFunction(double* coordinates, bool* neighbours,
                       EPosition position)
      : ShapeFunction(coordinates, neighbours, position) {}

  virtual double ComputeValue(double x, double y);
  ~EdgeTopShapeFunction() {}
};

class EdgeBotShapeFunction : public ShapeFunction {
public:
  EdgeBotShapeFunction(double* coordinates, bool* neighbours,
                       EPosition position)
      : ShapeFunction(coordinates, neighbours, position) {}

  virtual double ComputeValue(double x, double y);
  ~EdgeBotShapeFunction() {}
};

class EdgeRightShapeFunction : public ShapeFunction {
public:
  EdgeRightShapeFunction(double* coordinates, bool* neighbours,
                         EPosition position)
      : ShapeFunction(coordinates, neighbours, position) {}

  virtual double ComputeValue(double x, double y);
  ~EdgeRightShapeFunction() {}
};

class InteriorShapeFunction : public ShapeFunction {
public:
  InteriorShapeFunction(double* coordinates, bool* neighbours,
                        EPosition position)
      : ShapeFunction(coordinates, neighbours, position) {}

  virtual double ComputeValue(double x, double y);
  ~InteriorShapeFunction() {}
};
} // namespace D2
#endif
