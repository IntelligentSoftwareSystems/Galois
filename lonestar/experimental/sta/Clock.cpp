#include "Clock.h"

void Clock::print(std::ostream& os) {
  os << "create_clock -period " << period;
  os << " -name " << name;
  if (src) {
    os << " [get_ports " << src->name << "]";
  }
  os << std::endl;
}

void ClockEdge::print(std::ostream& os) {
  os << "(" << t << ", " << ((isRise) ? "r" : "f") << ")";
}
