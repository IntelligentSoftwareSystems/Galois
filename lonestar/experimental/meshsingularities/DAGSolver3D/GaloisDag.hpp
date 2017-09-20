#ifndef GALOIS_DAG_HPP
#define GALOIS_DAG_HPP

#include "galois/Galois.h"
#include "galois/Accumulator.h"
#include "galois/Bag.h"
#include "galois/CilkInit.h"
#include "galois/Timer.h"
#include "galois/runtime/TreeExec.h"


#include "Node.hpp"
#include "EquationSystem.hpp"

struct GaloisEliminationDivide
{

    GaloisEliminationDivide () {}

    template <typename C>
    void operator () (Node *node, C &ctx)
    {
        if (node->getLeft() != NULL && node->getRight() != NULL) {
            ctx.spawn (node->getLeft());
            ctx.spawn (node->getRight());
        }
    }
};

struct GaloisEliminationConquer
{
    GaloisEliminationConquer() {};

    void operator() (Node *node)
    {
        node->eliminate();
    }
};

struct GaloisBackwardSubstitutionDivide
{
    GaloisBackwardSubstitutionDivide () {}
    template
    <typename C>
    void operator () (Node *node, C &ctx)
    {
        node->bs();
        if (node->getLeft() != NULL && node->getRight() != NULL) {
            ctx.spawn(node->getLeft());
            ctx.spawn(node->getRight());
        }
    }
};

struct GaloisBackwardSubstitutionConquer
{
    void operator () (Node *n)
    {
        // empty
    }
};

struct GaloisAllocationDivide
{

    template <typename C>
    void operator () (Node *node, C &ctx)
    {
        if (node->getLeft() != NULL && node->getRight() != NULL) {
            ctx.spawn(node->getLeft());
            ctx.spawn(node->getRight());
        }
    }
};

struct GaloisAllocationConquer
{
    SolverMode mode;
    GaloisAllocationConquer(SolverMode m): mode(m) {}

    void operator() (Node *node)
    {
        node->allocateSystem(mode);
    }
};


void galoisAllocation(Node *node, SolverMode mode);
void galoisElimination (Node *node);
void galoisBackwardSubstitution(Node *node);

#endif
