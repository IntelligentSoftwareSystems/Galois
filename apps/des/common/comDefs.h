#ifndef _COM_DEFS_H
#define _COM_DEFS_H

#include <limits>
#include <string>
#include <algorithm>

typedef long long SimTime;

/** The Constant INFINITY_SIM_TIME. */
// const SimTime INFINITY_SIM_TIME = std::numeric_limits<SimTime>::max ();
// The above definition is bad because INFINITY_SIM_TIME + small_value will cause an overflow
// and the result is not INFINITY_SIM_TIME any more
const SimTime INFINITY_SIM_TIME = (1 << 30);


std::string  toLowerCase (std::string str);
#endif
