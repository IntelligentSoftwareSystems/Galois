#include "galois/DoAllWrap.h"

namespace galois {

void setDoAllImpl (const DoAllTypes& type) {
  doAllKind = type;
}

DoAllTypes getDoAllImpl (void) {
  return doAllKind;
}

} // end namespace galois

