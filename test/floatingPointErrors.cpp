#include <boost/rational.hpp>

#include <random>
#include <iostream>

#include <cstdio>

using Rational = boost::rational<size_t>;

void multiplyTest(const double mpcand, const double mplier, const double ans) {

  double lim = mplier / 100.0;
  assert(lim >= 1.0);

  std::mt19937 eng;
  eng.seed(0);

  std::uniform_real_distribution<double> dist(0.0, lim);

  double remainMplier = mplier;

  double computed = 0.0;

  while (remainMplier > 0.0) {

    double partial = dist(eng);

    if (partial > remainMplier) {
      partial = remainMplier;
    }

    remainMplier -= partial;

    computed += mpcand * partial;
  }

  std::printf("Error in multiplication with doubles = %g\n", (ans - computed));
}

void multiplyTestRational(const Rational& mpcand, const Rational& mplier,
                          const Rational& ans) {

  size_t lim = boost::rational_cast<size_t>(mplier / Rational(100));

  std::mt19937 eng;
  eng.seed(0);

  std::uniform_int_distribution<size_t> dist(1, lim);

  Rational remainMplier = mplier;

  Rational computed(0);

  while (remainMplier > Rational(0)) {

    Rational partial(dist(eng), lim);

    if (partial > remainMplier) {
      partial = remainMplier;
    }

    // std::cout << "Rational partial mpcand: " << partial << std::endl;

    remainMplier -= partial;

    computed += mpcand * partial;
  }

  std::cout << "Error in multiplication with Rational: " << (ans - computed)
            << std::endl;
}

void rationalConversionError(double fpVal) {

  static const unsigned SIGNIFICANT_BITS = 40;

  size_t q = (size_t(1) << SIGNIFICANT_BITS);
  size_t p = size_t(fpVal * q);

  Rational r(p, q);

  std::printf("Conversion error = %g\n",
              (fpVal - boost::rational_cast<double>(r)));
}

int main() {

  Rational x(1, 3);

  multiplyTest(0.125, 1000.0, 125.0);

  multiplyTestRational(Rational(125, 1000), Rational(1000), Rational(125));

  rationalConversionError(sqrt(2.0));
  rationalConversionError(sqrt(3.0));
  rationalConversionError(sqrt(1000.0));
  rationalConversionError(sqrt(100000.0));
  rationalConversionError(sqrt(15485867)); // prime number

  return 0;
}
