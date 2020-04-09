#include <cmath>
#include <cstdio>
#include "Utils.h"

#ifndef PI
#define PI 3.14159265358979323846
#endif

bool Utils::is_lesser(double a, double b) { return a - b < -2 * EPSILON; }

bool Utils::is_greater(double a, double b) { return a - b > 2 * EPSILON; }

bool Utils::equals(const double a, const double b) {
  return fabs(a - b) < EPSILON;
}

double Utils::floor2(double a) {
  double b = (int)a;
  if (!(!is_greater(b, a) && is_greater(b + 1, a))) {
    ++b;
  }
  return b;
}

double Utils::ceil2(double a) { return floor2(a) + 1; }

void Utils::change_bytes_order(uint16_t* var_ptr) {
  uint16_t tmp = *var_ptr;
  tmp <<= 8;
  (*var_ptr) >>= 8;
  (*var_ptr) |= tmp;
}

void Utils::swap_if_required(double* should_be_lower,
                             double* should_be_bigger) {
  if ((*should_be_lower) > (*should_be_bigger)) {
    double tmp          = (*should_be_lower);
    (*should_be_lower)  = (*should_be_bigger);
    (*should_be_bigger) = tmp;
  }
}

size_t Utils::gcd(size_t a, size_t b) {
  do {
    if (b > a) {
      size_t tmp = a;
      a          = b;
      b          = tmp;
    }
    a -= b;
  } while (a != 0);
  return b;
}

double Utils::d2r(double degrees) { return degrees * PI / 180; }

double Utils::r2d(double radians) { return radians * 180 / PI; }

void Utils::shift(int from, int to, size_t* array) {
  for (int i = to; i > from; --i) {
    array[i] = array[i - 1];
  }
}

std::pair<double, double> Utils::convertToUtm(double latitude, double longitude,
                                              Map& map) {
  long zone;
  char hemisphere;
  double easting;
  double northing;
  if (Convert_Geodetic_To_UTM(d2r(latitude), d2r(longitude), &zone, &hemisphere,
                              &easting, &northing)) {
    fprintf(stderr, "Error during conversion to UTM.\n");
    exit(13);
  }
  map.setZone(zone);
  map.setHemisphere(hemisphere);
  return std::pair<double, double>(easting, northing);
}