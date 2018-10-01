#include "Clock.h"

void Clock::print(std::ostream& os) {
  os << "  Clock " << name << ":" << std::endl;
  os << "    period = " << period << std::endl;
  os << "    src pin = " << ((src) ? src->name : "(virtual)") << std::endl;
  os << "    wave from = { ";
  for (auto& i: waveform) {
    i.print(os);
    os << " ";
  }
  os << "}" << std::endl;
}

void ClockEdge::print(std::ostream& os) {
  os << "(" << t << ", " << ((isRise) ? "r" : "f") << ")";
}
