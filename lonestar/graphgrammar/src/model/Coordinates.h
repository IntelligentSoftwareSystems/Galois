#ifndef GALOIS_COORDINATES_H
#define GALOIS_COORDINATES_H

#include <ostream>
#include "../utils/utils.h"
#include "Map.h"

/**
 * Container for x, y, and z coordinates and various utitlity functions on the
 * coordinates
 */
class Coordinates {
private:
  double x;
  double y;
  double z;

public:

  //! empty constructor: x, y, z initialized to default double
  Coordinates() = default;

  //! Initialize x/y as specified, z = 0
  Coordinates(double x, double y) : x(x), y(y), z(0.) {}

  //! Initialize all 3 as specified
  Coordinates(double x, double y, double z) : x(x), y(y), z(z) {}

  //! Determine z from given x, y based on the provided Map
  Coordinates(double x, double y, Map& map)
      : x(x), y(y), z(map.get_height(x, y)) {}

  //! x, y in the pair, height given by Map
  Coordinates(std::pair<double, double> coords, Map& map)
      : x(coords.first), y(coords.second),
        z(map.get_height(coords.first, coords.second)) {}

  // what follows are self explanatory get/set functions

  double getX() const { return x; }

  void setX(double x) { Coordinates::x = x; }

  double getY() const { return y; }

  void setY(double y) { Coordinates::y = y; }

  double getZ() const { return z; }

  void setZ(double z) { Coordinates::z = z; }

  void setXYZ(double x, double y, double z) {
    Coordinates::x = x;
    Coordinates::y = y;
    Coordinates::z = z;
  }

  //! Get 2D or 3D distances given another set of coordinates
  double dist(const Coordinates& rhs, bool version2D) const {
    if (version2D) {
      return dist2D(rhs);
    } else {
      return dist3D(rhs);
    }
  }

  //! Take z into account for distance
  double dist3D(const Coordinates& rhs) const {
    return sqrt(pow(x - rhs.x, 2) + pow(y - rhs.y, 2) + pow(z - rhs.z, 2));
  }


  //! Distance of just x/y coordinates
  double dist2D(const Coordinates& rhs) const {
    return sqrt(pow(x - rhs.x, 2) + pow(y - rhs.y, 2));
  }

  bool isXYequal(const Coordinates& rhs) {
    return equals(x, rhs.x) && equals(y, rhs.y);
  }

  std::string toString() const {
    return std::to_string(x) + " " + std::to_string(y) + " " +
           std::to_string(z);
  }


  //! element wise add of x,y,z
  Coordinates operator+(const Coordinates& rhs) const {
    return Coordinates{x + rhs.x, y + rhs.y, z + rhs.z};
  }

  //! element wise subtract of x,y,z
  Coordinates operator-(const Coordinates& rhs) const {
    return Coordinates{x - rhs.x, y - rhs.y, z - rhs.z};
  }


  //! element wise multiply of x,y,z
  Coordinates operator*(double rhs) const {
    return Coordinates{x * rhs, y * rhs, z * rhs};
  }


  //! element wise divide of x,y,z
  Coordinates operator/(double rhs) const {
    return Coordinates{x / rhs, y / rhs, z / rhs};
  }


  //! element wise equality check
  bool operator==(const Coordinates& rhs) const {
    return equals(x, rhs.x) && equals(y, rhs.y) && equals(z, rhs.z);
  }


  //! element wise inequality check
  bool operator!=(const Coordinates& rhs) const { return !(rhs == *this); }

  //! Less than check; checks x, y, z in that order
  bool operator<(const Coordinates& rhs) const {
    if (less(x, rhs.x))
      return true;
    if (less(rhs.x, x))
      return false;
    if (less(y, rhs.y))
      return true;
    if (less(rhs.y, y))
      return false;
    return less(z, rhs.z);
  }

  bool operator>(const Coordinates& rhs) const { return rhs < *this; }

  bool operator<=(const Coordinates& rhs) const { return !(rhs < *this); }

  bool operator>=(const Coordinates& rhs) const { return !(*this < rhs); }
};

#endif // GALOIS_COORDINATES_H
