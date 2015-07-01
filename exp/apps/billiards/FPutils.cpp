#include "FPutils.h"

const double FPutils::EPSILON = 1.0e-15;
const unsigned FPutils::TRUCATE_FRACTION_BITS = 20;
const uint64_t FPutils::PRECISION_64 = (uint64_t(1) << FPutils::TRUCATE_FRACTION_BITS);
const double FPutils::TRUNCATE_PRECISION = FPutils::PRECISION_64;
