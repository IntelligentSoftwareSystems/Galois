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
