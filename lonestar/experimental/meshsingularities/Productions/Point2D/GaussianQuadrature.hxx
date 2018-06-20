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

#ifndef __GAUSSIANQUADRATURE_2D_H_INCLUDED__
#define __GAUSSIANQUADRATURE_2D_H_INCLUDED__

#include "DoubleArgFunction.hxx"
namespace D2 {
class GaussianQuadrature {
private:
  static double revertNormalization(double x, double lower, double upper) {
    return (lower + upper) / 2.0 + (upper - lower) / 2.0 * x;
  }

public:
  static double
  definiteDoubleIntegral(double lower1, double upper1, double lower2,
                         double upper2,
                         IDoubleArgFunction* functionToIntegrate) {
    static const double roots[4]   = {-0.86113, -0.33998, 0.33998, 0.86113};
    static const double weights[4] = {0.34785, 0.65214, 0.65214, 0.34785};
    double integral                = 0;
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 4; j++) {

        double v = weights[i] * weights[j] *
                   functionToIntegrate->ComputeValue(
                       revertNormalization(roots[i], lower1, upper1),
                       revertNormalization(roots[j], lower2, upper2));
        integral += v;
      }
    }

    return integral * (upper1 - lower1) * (upper2 - lower2) / 4.0;
  }
};
} // namespace D2
#endif
