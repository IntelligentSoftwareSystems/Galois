#ifndef __GAUSSIANQUADRATURE_3D_H_INCLUDED__
#define __GAUSSIANQUADRATURE_3D_H_INCLUDED__

#include "TripleArgFunction.hxx"
namespace D3 {
class GaussianQuadrature {
private:
  static double revertNormalization(double x, double lower, double upper) {
    return (lower + upper) / 2.0 + (upper - lower) / 2.0 * x;
  }

public:
  static double
  definiteTripleIntegral(double lower1, double upper1, double lower2,
                         double upper2, double lower3, double upper3,
                         ITripleArgFunction* functionToIntegrate) {
    static const double roots[4]   = {-0.86113, -0.33998, 0.33998, 0.86113};
    static const double weights[4] = {0.34785, 0.65214, 0.65214, 0.34785};
    double integral                = 0;
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 4; j++) {
        for (int k = 0; k < 4; k++) {
          double v = weights[i] * weights[j] * weights[k] *
                     functionToIntegrate->ComputeValue(
                         revertNormalization(roots[i], lower1, upper1),
                         revertNormalization(roots[j], lower2, upper2),
                         revertNormalization(roots[k], lower3, upper3));
          integral += v;
        }
      }
    }

    return integral * (upper1 - lower1) * (upper2 - lower2) *
           (upper3 - lower3) / 8.0;
  }
};
} // namespace D3
#endif
