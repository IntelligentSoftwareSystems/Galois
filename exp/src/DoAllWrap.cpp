#include "Galois/DoAllWrap.h"

namespace Galois {

cll::opt<DoAllTypes> doAllKind (
    cll::desc ("DoAll Implementation"),
    cll::values (
      clEnumVal (GALOIS, "GALOIS"),
      clEnumVal (GALOIS_STEAL, "GALOIS_STEAL"),
      clEnumVal (COUPLED, "COUPLED"),
      clEnumVal (CILK, "CILK"),
      clEnumVal (OPENMP, "OPENMP"),
      clEnumValEnd),
    cll::init (GALOIS));

} // end namespace Galois

