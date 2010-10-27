#include <cassert>
#include "Galois/Galois.h"

__thread GaloisRuntime::SimpleRuntimeContext* GaloisRuntime::thread_cnx;
