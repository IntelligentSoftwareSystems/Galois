/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
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

    remainMplier -= partial;

    computed += mpcand * partial;
  }

  std::cout << "Error in multiplication with Rational: " << (ans - computed)
            << "\n";
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
  multiplyTest(0.125, 1000.0, 125.0);

  multiplyTestRational(Rational(125, 1000), Rational(1000), Rational(125));

  rationalConversionError(boost::rational_cast<double>(Rational(1, 3)));

  rationalConversionError(sqrt(2.0));
  rationalConversionError(sqrt(3.0));
  rationalConversionError(sqrt(1000.0));
  rationalConversionError(sqrt(100000.0));
  rationalConversionError(sqrt(15485867)); // prime number

  return 0;
}
