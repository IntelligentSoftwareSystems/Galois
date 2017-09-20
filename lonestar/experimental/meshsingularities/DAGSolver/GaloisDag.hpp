#ifndef GALOIS_DAG_HPP
#define GALOIS_DAG_HPP

#include "Galois/Galois.h"
#include "Galois/Accumulator.h"
#include "Galois/Bag.h"
#include "Galois/CilkInit.h"
#include "Galois/Timer.h"
#include "Galois/Runtime/TreeExec.h"


#include "Node.hpp"
#include "EquationSystem.h"

struct GaloisElimination: public galois::Runtime::TreeTaskBase
{
    Node *node;

    GaloisElimination (Node *_node):
        galois::Runtime::TreeTaskBase (),
        node (_node)
    {}

    virtual void operator () (galois::Runtime::TreeTaskContext& ctx)
    {
        if (node->getLeft() != NULL && node->getRight() != NULL) {
            GaloisElimination left {node->getLeft()};

            GaloisElimination right {node->getRight()};
            ctx.spawn (left);
            ctx.spawn (right);

            ctx.sync ();
        }
        node->eliminate();

    }
};

struct GaloisBackwardSubstitution: public galois::Runtime::TreeTaskBase
{
    Node *node;

    GaloisBackwardSubstitution (Node *_node):
        galois::Runtime::TreeTaskBase(),
        node(_node)
    {}

    virtual void operator () (galois::Runtime::TreeTaskContext &ctx)
    {
        node->bs();
        if (node->getLeft() != NULL && node->getRight() != NULL) {
            // change to galois::for_each (scales better)
            GaloisBackwardSubstitution left { node->getLeft() };
            GaloisBackwardSubstitution right { node->getRight() };

            ctx.spawn(left);
            ctx.spawn(right);
            ctx.sync();
        }
    }
};

struct GaloisAllocation: public galois::Runtime::TreeTaskBase
{
    Node *node;
    SolverMode mode;
    GaloisAllocation (Node *_node, SolverMode _mode):
        galois::Runtime::TreeTaskBase(),
        node(_node), mode(mode)
    {}

    virtual void operator () (galois::Runtime::TreeTaskContext &ctx)
    {
        node->allocateSystem(mode);
        if (node->getLeft() != NULL && node->getRight() != NULL) {
            GaloisAllocation left { node->getLeft(), mode };

            GaloisAllocation right { node->getRight(), mode };
            ctx.spawn(left);

            ctx.spawn(right);

            ctx.sync();
        }
    }
};

void galoisAllocation(Node *node, SolverMode mode);
void galoisElimination (Node *node);
void galoisBackwardSubstitution(Node *node);

#endif
