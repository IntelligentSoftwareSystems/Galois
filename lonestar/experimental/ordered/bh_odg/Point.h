#ifndef BH_POINT_H
#define BH_POINT_H

namespace bh {

bool checkRelativeError(const double ref, const double obs) {
  const double THRESHOLD = 1.0e-10;
  return fabs((ref - obs) / ref) < THRESHOLD;
}

struct Point {
  double x, y, z;
  Point() : x(0.0), y(0.0), z(0.0) {}
  Point(double _x, double _y, double _z) : x(_x), y(_y), z(_z) {}
  explicit Point(double v) : x(v), y(v), z(v) {}

  double operator[](const int index) const {
    switch (index) {
    case 0:
      return x;
    case 1:
      return y;
    case 2:
      return z;
    }
    assert(false && "index out of bounds");
    abort();
  }

  double& operator[](const int index) {
    switch (index) {
    case 0:
      return x;
    case 1:
      return y;
    case 2:
      return z;
    }
    assert(false && "index out of bounds");
    abort();
  }

  bool operator==(const Point& other) {
    if (x == other.x && y == other.y && z == other.z)
      return true;
    return false;
  }

  bool operator!=(const Point& other) { return !operator==(other); }

  Point& operator+=(const Point& other) {
    x += other.x;
    y += other.y;
    z += other.z;
    return *this;
  }

  Point& operator-=(const Point& other) {
    x -= other.x;
    y -= other.y;
    z -= other.z;
    return *this;
  }

  Point& operator*=(double value) {
    x *= value;
    y *= value;
    z *= value;
    return *this;
  }

  double mag() const { return sqrt(x * x + y * y + z * z); }

  friend std::ostream& operator<<(std::ostream& os, const Point& p) {
    os << "(" << p[0] << "," << p[1] << "," << p[2] << ")";
    return os;
  }

  friend bool checkRelativeError(const Point& ref, const Point& obs) {
    // Point tmp (ref);
    // tmp -= obs;
    // return (tmp.mag () / ref.mag ()) < THRESHOLD;
    return checkRelativeError(ref.x, obs.x) &&
           checkRelativeError(ref.y, obs.y) && checkRelativeError(ref.z, obs.z);
  }
};

} // end namespace bh
#endif // BH_POINT_H
