#include "AsyncTimingArcSet.h"

void AsyncTimingArcCollection::setup() {
  for (auto& i: v.modules) {
    modules[i.second] = new AsyncTimingArcSet;
  }
}

void AsyncTimingArcCollection::clear() {
  for (auto& i: modules) {
    delete i.second;
  }
}
