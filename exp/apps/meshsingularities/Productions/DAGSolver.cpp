#include "DynamicLib.h"
#include "ExternalMesh/Analysis.hpp"
#include "ExternalMesh/Node.hpp"
#include "ExternalMesh/Element.hpp"
#include "ExternalMesh/Mesh.hpp"

#include "EquationSystem.h"

#include "Galois/Galois.h"
#include "Galois/Accumulator.h"
#include "Galois/Bag.h"
#include "Galois/CilkInit.h"
#include "Galois/GaloisUnsafe.h"
#include "Galois/Statistic.h"
#include "Galois/Runtime/TreeExec.h"

#include "Lonestar/BoilerPlate.h"

#include "llvm/Support/CommandLine.h"

#include <sys/time.h>

enum Schedulers
{
    OLD,
    CILK,
    GALOIS_DAG
};

const char* const name = "DAGSolver";
const char* const desc = "Solver for FEM with singularities on meshes";
const char* const url = NULL;

namespace cll = llvm::cl;

static cll::opt<std::string> prodlib("prodlib", cll::desc("Shared library with productions code"),
                                     cll::init("./pointproductions.so"));

static cll::opt<std::string> treefile("treefile", cll::desc("File with tree definition"),
                                     cll::init("./tree.txt"));

static cll::opt<Schedulers> scheduler("scheduler", cll::desc("Scheduler"),
                                      cll::values(clEnumVal(CILK, "Cilk-based"),
                                                  clEnumVal(GALOIS_DAG, "Galois-DAG"),
                                                  clEnumValEnd), cll::init(CILK));
using namespace std;

#ifdef HAVE_CILK
void cilk_alloc_tree(Node *n)
{
    n->allocateSystem();
    if (n->getRight() != NULL && n->getLeft() != NULL) {
        cilk_spawn cilk_alloc_tree(n->getLeft());
        cilk_spawn cilk_alloc_tree(n->getRight());
        cilk_sync;
        //n->mergeProduction = (void(*)(double**, double*, double**, double*, double**, double*))
        //        lib->resolvSymbol(n->getProduction());
    } else {
        //n->preprocessProduction = (void(*)(double**, double*, double**, double*))lib->resolvSymbol(n->getProduction());
    }
}

void cilk_do_elimination(Node *n)
{
    if (n->getRight() != NULL && n->getLeft() != NULL) {
        cilk_spawn cilk_do_elimination(n->getLeft());
        cilk_spawn cilk_do_elimination(n->getRight());
        cilk_sync;
    }
    n->eliminate();
}

void cilk_do_backward_substitution(Node *n)
{
    n->bs();
    if (n->getRight() != NULL && n->getLeft() != NULL) {
        cilk_spawn cilk_do_backward_substitution(n->getLeft());
        cilk_spawn cilk_do_backward_substitution(n->getRight());
        cilk_sync;
    }
}

#endif // HAVE_CILK


struct GaloisElimination: public Galois::Runtime::TreeTaskBase
{
    Node *node;

    GaloisElimination (Node *_node):
        Galois::Runtime::TreeTaskBase (),
        node (_node)
    {}

    virtual void operator () (Galois::Runtime::TreeTaskContext& ctx)
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

struct GaloisBackwardSubstitution: public Galois::Runtime::TreeTaskBase
{
    Node *node;

    GaloisBackwardSubstitution (Node *_node):
        Galois::Runtime::TreeTaskBase(),
        node(_node)
    {}

    virtual void operator () (Galois::Runtime::TreeTaskContext &ctx)
    {
        node->bs();
        if (node->getLeft() != NULL && node->getRight() != NULL) {
            // change to Galois::for_each (scales better)
            GaloisBackwardSubstitution left { node->getLeft() };
            GaloisBackwardSubstitution right { node->getRight() };

            ctx.spawn(left);

            ctx.spawn(right);
            ctx.sync();
        }
    }
};

struct GaloisAllocation: public Galois::Runtime::TreeTaskBase
{
    Node *node;

    GaloisAllocation (Node *_node):
        Galois::Runtime::TreeTaskBase(),
        node(_node)
    {}

    virtual void operator () (Galois::Runtime::TreeTaskContext &ctx)
    {
        node->allocateSystem();
        if (node->getLeft() != NULL && node->getRight() != NULL) {
            GaloisAllocation left { node->getLeft() };

            GaloisAllocation right { node->getRight() };
            ctx.spawn(left);

            ctx.spawn(right);

            ctx.sync();

            //n->mergeProduction = (void(*)(double**, double*, double**, double*, double**, double*))
            //        lib->resolvSymbol(n->getProduction());
        //} else {
            //n->preprocessProduction = (void(*)(double**, double*, double**, double*))lib->resolvSymbol(n->getProduction());
        }
    }
};

void galoisAllocation(Node *node)
{
    GaloisAllocation root {node};
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

void print_time(char *msg, timeval *t1, timeval *t2)
{
    printf("%s: %f\n", msg, t2->tv_sec-t1->tv_sec + (t2->tv_usec-t1->tv_usec)/1e+6);
}

int main(int argc, char ** argv)
{
    LonestarStart(argc, argv, name, desc, url);
    struct timeval t1, t2;

    //DynamicLib *lib = new DynamicLib(prodlib);
    //lib->load();

    printf("Singularity solver - run info:\n");
    printf("\tmesh file: %s\n", treefile.c_str());

    Mesh *m  = Mesh::loadFromFile(treefile.c_str());
    if (m == NULL) {
        printf("Could not load the mesh. Exiting.\n");
        exit(1);
    }

    printf("Analysis part.\n");
    gettimeofday(&t1, NULL);
    Analysis::enumerateDOF(m);
    gettimeofday(&t2, NULL);
    print_time("\tDOF enumeration", &t1, &t2);

    gettimeofday(&t1, NULL);
    Analysis::doAnalise(m);
    gettimeofday(&t2, NULL);
    print_time("\tanalysis", &t1, &t2);

    printf("\tnumber of elements: %lu\n", m->getElements().size());
    printf("\tproblem size (dofs): %lu\n", m->getDofs());

    printf("Solving part.\n");

    if (scheduler == GALOIS_DAG) {
        gettimeofday(&t1, NULL);
        galoisAllocation(m->getRootNode());
        gettimeofday(&t2, NULL);
        print_time("\tallocation", &t1, &t2);

        gettimeofday(&t1, NULL);
        galoisElimination(m->getRootNode());
        gettimeofday(&t2, NULL);
        print_time("\tfactorization", &t1, &t2);

        gettimeofday(&t1, NULL);
        galoisBackwardSubstitution(m->getRootNode());
        gettimeofday(&t2, NULL);
        print_time("\tsolution", &t1, &t2);

    } else if (scheduler == CILK) {
#ifdef HAVE_CILK
        Galois::CilkInit();
        gettimeofday(&t1, NULL);
        cilk_alloc_tree(m->getRootNode());
        gettimeofday(&t2, NULL);
        print_time("\tallocation", &t1, &t2);

        gettimeofday(&t1, NULL);
        cilk_do_elimination(m->getRootNode());
        gettimeofday(&t2, NULL);
        print_time("\tfactorization", &t1, &t2);

        gettimeofday(&t1, NULL);
        cilk_do_backward_substitution(m->getRootNode());
        gettimeofday(&t2, NULL);
        print_time("\tsolution", &t1, &t2);

#else
        printf("CILK is not supported.\n");
#endif
    }

    delete m;
    //delete lib;

    return 0;
}
