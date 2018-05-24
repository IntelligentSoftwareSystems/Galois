#include "GeomUtils.h"
#include "FPutils.h"

Vec2 operator + (const Vec2& v1, const Vec2& v2) {

  Vec2 sum (v1);
  sum += v2;
  return sum;
}

Vec2 operator - (const Vec2& v1, const Vec2& v2) {

  Vec2 diff (v1);
  diff -= v2;
  return diff;
}

Vec2 operator / (const Vec2& v1, const FP& s) {

  Vec2 sv (v1);
  sv /= s;
  return sv;
}

Vec2 operator * (const Vec2& v1, const FP& s) {

  Vec2 sv (v1);
  sv *= s;
  return sv;
}

Vec2 operator * (const FP& s, const Vec2& v1) {
  return v1 * s;
}

std::ostream& operator << (std::ostream& out, const Vec2& v) {
  return (out << v.str ());
}


