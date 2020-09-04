
#ifndef TERGEN_UTILS_H
#define TERGEN_UTILS_H

#include <cstdint>
#include <cstdlib>
#include <utility>
#include "../libmgrs/utm.h"
#include "../model/Map.h"

class Utils {
public:
  constexpr static const double EPSILON = 1e-6;

  static bool is_lesser(double a, double b);

  static bool is_greater(double a, double b);

  static bool equals(const double a, const double b);

  static double floor2(double a);

  static double ceil2(double a);

  static void change_bytes_order(uint16_t* var_ptr);

  static void swap_if_required(double* should_be_lower,
                               double* should_be_bigger);

  static size_t gcd(size_t a, size_t b);

  static double d2r(double degrees);

  static double r2d(double radians);

  static void shift(int from, int to, size_t* array);

  static std::pair<double, double> convertToUtm(double latitude,
                                                double longitude, Map& map);
};

#endif // TERGEN_UTILS_H
