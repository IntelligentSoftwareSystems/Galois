#include "GaloisDag.hpp"

void galoisAllocation(Node *node, SolverMode mode)
{
    GaloisAllocation root {node, mode};
    Galois::Runtime::for_each_ordered_tree_generic(root, "alloc-gen");
}

void galoisElimination (Node *node)
{
    GaloisElimination root {node};
    Galois::Runtime::for_each_ordered_tree_generic (root, "elim-gen");
}

void galoisBackwardSubstitution(Node *node)
{
    GaloisBackwardSubstitution root {node};
    Galois::Runtime::for_each_ordered_tree_generic(root, "bs-gen");
}
