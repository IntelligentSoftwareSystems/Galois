#ifndef CILK_HPP
#define CILK_HPP

#include "Galois/CilkInit.h"

#include "Node.hpp"
#include "EquationSystem.hpp"

void cilk_alloc_tree(Node *n, SolverMode mode);
void cilk_do_elimination(Node *n);
void cilk_do_backward_substitution(Node *n);

#endif
