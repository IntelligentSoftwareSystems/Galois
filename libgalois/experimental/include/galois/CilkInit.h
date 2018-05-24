#ifndef GALOIS_CILK_INIT_H
#define GALOIS_CILK_INIT_H

#ifdef HAVE_CILK
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>

#else
#define cilk_for for
#define cilk_spawn
#define cilk_sync
#endif

namespace galois {

  void CilkInit (void);

} // end namespace galois
#endif // GALOIS_CILK_INIT_H
