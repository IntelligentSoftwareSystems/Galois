#include "FPutils.h"
#include "GeomUtils.h"

const FP FPutils::EPSILON = double (1.0 / (1 << 20));
const FP FPutils::ERROR_LIMIT = double (1.0 / (1 << 20));

bool FPutils::almostEqual (const Vec2& v1, const Vec2& v2) {
  return almostEqual (v1.getX (), v2.getX ()) && almostEqual (v1.getY (), v2.getY ());
}

bool FPutils::checkError (const Vec2& original, const Vec2& measured, bool useAssert) {

  return checkError (original.getX (), measured.getX (), useAssert) 
    && checkError (original.getY (), measured.getY (), useAssert);
}

const Vec2& FPutils::truncate (const Vec2& vec) { return vec; }
