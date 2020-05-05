/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 */

#include "CILK.hpp"

void cilk_alloc_tree(Node* n, SolverMode mode) {
  n->allocateSystem(mode);
  if (n->getRight() != NULL && n->getLeft() != NULL) {
#ifdef HAVE_CILK
    cilk_spawn
#endif
        cilk_alloc_tree(n->getLeft(), mode);
#ifdef HAVE_CILK
    cilk_spawn
#endif
        cilk_alloc_tree(n->getRight(), mode);
#ifdef HAVE_CILK
    cilk_sync;
#endif
  }
}

void cilk_do_elimination(Node* n) {
  if (n->getRight() != NULL && n->getLeft() != NULL) {
#ifdef HAVE_CILK
    cilk_spawn
#endif
        cilk_do_elimination(n->getLeft());
#ifdef HAVE_CILK
    cilk_spawn
#endif
        cilk_do_elimination(n->getRight());
#ifdef HAVE_CILK
    cilk_sync;
#endif
  }
  n->eliminate();
}

void cilk_do_backward_substitution(Node* n) {
  n->bs();
  if (n->getRight() != NULL && n->getLeft() != NULL) {
#ifdef HAVE_CILK
    cilk_spawn
#endif
        cilk_do_backward_substitution(n->getLeft());
#ifdef HAVE_CILK
    cilk_spawn
#endif
        cilk_do_backward_substitution(n->getRight());
#ifdef HAVE_CILK
    cilk_sync;
#endif
  }
}
