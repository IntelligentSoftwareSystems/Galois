#include "CILK.hpp"

void cilk_alloc_tree(Node *n, SolverMode mode)
{
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

void cilk_do_elimination(Node *n)
{
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

void cilk_do_backward_substitution(Node *n)
{
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
