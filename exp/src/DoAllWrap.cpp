#include "Galois/DoAllWrap.h"

namespace Galois {

cll::opt<DoAllTypes> doAllKind (
    cll::desc ("DoAll Implementation"),
    cll::values (
      clEnumVal (DOALL_GALOIS, "DOALL_GALOIS"),
      clEnumVal (DOALL_GALOIS_STEAL, "DOALL_GALOIS_STEAL"),
      clEnumVal (DOALL_GALOIS_FOREACH, "DOALL_GALOIS_FOREACH"),
      clEnumVal (DOALL_COUPLED, "DOALL_COUPLED"),
      clEnumVal (DOALL_CILK, "DOALL_CILK"),
      clEnumVal (DOALL_OPENMP, "DOALL_OPENMP"),
      clEnumValEnd),
    cll::init (DOALL_COUPLED));

void setDoAllImpl (const DoAllTypes& type) {
  doAllKind = type;
}

DoAllTypes getDoAllImpl (void) {
  return doAllKind;
}

} // end namespace Galois

