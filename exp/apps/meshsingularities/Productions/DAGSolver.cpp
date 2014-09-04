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
#include "Galois/Statistic.h"
#include "Galois/Runtime/TreeExec.h"

#include "Lonestar/BoilerPlate.h"

#include "llvm/Support/CommandLine.h"

#include <sys/time.h>

#ifdef WITH_PAPI
#include "papi.h"
#endif

enum Schedulers
{
    OLD,
    CILK,
    GALOIS_DAG,
    SEQ
};

enum Rotator
{
    TRUE,
    FALSE
};

const char* const name = "DAGSolver";
const char* const desc = "Solver for FEM with singularities on meshes";
const char* const url = NULL;

namespace cll = llvm::cl;

static cll::opt<std::string> prodlib("prodlib", cll::desc("Shared library with productions code"),
                                     cll::init("./pointproductions.so"));

static cll::opt<std::string> treefile("treefile", cll::desc("File with tree definition"),
                                     cll::init(""));

static cll::opt<std::string> matrixfile("matrixfile", cll::desc("File with frontal matrices"),
                                        cll::init(""));

static cll::opt<bool> debug("debug", cll::desc("Debug mode"), cll::init(false));

static cll::opt<Schedulers> scheduler("scheduler", cll::desc("Scheduler"),
                                      cll::values(
#ifdef HAVE_CILK
                                          clEnumVal(CILK, "Cilk-based"),
#endif
                                          clEnumVal(GALOIS_DAG, "Galois-DAG"),
                                          clEnumVal(SEQ, "Sequential"),
                                          clEnumValEnd), cll::init(CILK));

static cll::opt<Rotator> rotation("rotation", cll::desc("Rotation"),
                                      cll::values(clEnumVal(TRUE, "true"),
                                                  clEnumVal(FALSE, "false"),
                                                  clEnumValEnd), cll::init(FALSE));

#ifdef WITH_PAPI
static cll::opt<bool> perfcounters("perfcounters", cll::desc("Enable performance counters"),
                                   cll::init(false));
#endif

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
        //Analysis::debugNode(node);
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

void seqAllocation(Node *node)
{
    node->allocateSystem();
    if (node->getLeft() != NULL && node->getRight() != NULL) {
        seqAllocation(node->getLeft());
        seqAllocation(node->getRight());
    }
}

void seqElimination(Node *node)
{
    if (node->getLeft() != NULL && node->getRight() != NULL) {
        seqElimination(node->getLeft());
        seqElimination(node->getRight());
    }
    node->eliminate();
}

void seqBackwardSubstitution(Node *node)
{
    node->bs();
    if (node->getLeft() != NULL && node->getRight() != NULL) {
        seqBackwardSubstitution(node->getLeft());
        seqBackwardSubstitution(node->getRight());
    }
}

void print_time(char *msg, timeval *t1, timeval *t2)
{
    printf("%s: %f\n", msg, ((t2->tv_sec-t1->tv_sec)*1000000 + (t2->tv_usec-t1->tv_usec))/1000000.0);
}

int main(int argc, char ** argv)
{
    LonestarStart(argc, argv, name, desc, url);
    struct timeval t1, t2;

#ifdef WITH_PAPI
    bool papi_supported = true;
    int events[7] = {PAPI_FP_OPS,
                    PAPI_TOT_INS,
                    PAPI_BR_INS,
                    PAPI_LD_INS,
                    PAPI_SR_INS,
                    PAPI_L1_DCM,
                    PAPI_L2_TCM};
    long long values[7] = {0,};

    int papi_err;
    if (PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT) {
        fprintf(stderr, "PAPI is unsupported.\n");
        papi_supported = false;
        }

    if (PAPI_num_counters() < 7) {
        fprintf(stderr, "PAPI is unsupported.\n");
        papi_supported = false;
    }
#endif
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

    //tree rotation
    if (rotation == TRUE){
        printf("Tree size %d\n", m->getRootNode()->treeSize());   // DEBUG
        gettimeofday(&t1, NULL);
        Analysis::rotate(m->getRootNode(), NULL, m);
        gettimeofday(&t2, NULL);
        printf("Tree size %d\n", m->getRootNode()->treeSize());   // DEBUG
        print_time("\tTree rotation", &t1, &t2);
    }
    
    gettimeofday(&t1, NULL);
    Analysis::doAnalise(m);
    gettimeofday(&t2, NULL);
    print_time("\tanalysis", &t1, &t2);

    printf("\tnumber of elements: %lu\n", m->getElements().size());
    printf("\tproblem size (dofs): %lu\n", m->getDofs());
    if (debug) {
        Analysis::printTree(m->getRootNode());

        for (Element *e : m->getElements()) {
            Analysis::printElement(e);
        }
    }

    printf("Solving part.\n");

    printf("Root size: %d\n", m->getRootNode()->getDofs().size());

    if (scheduler == GALOIS_DAG) {
        gettimeofday(&t1, NULL);
        galoisAllocation(m->getRootNode());
        gettimeofday(&t2, NULL);
        print_time("\tallocation", &t1, &t2);

#ifdef WITH_PAPI
        if (papi_supported) {
            if ((papi_err = PAPI_start_counters(events, 7)) != PAPI_OK) {
                fprintf(stderr, "Could not start counters: %s\n", PAPI_strerror(papi_err));
            }
        }
#endif
        gettimeofday(&t1, NULL);
        galoisElimination(m->getRootNode());
        gettimeofday(&t2, NULL);
        print_time("\tfactorization", &t1, &t2);

        gettimeofday(&t1, NULL);
        galoisBackwardSubstitution(m->getRootNode());
        gettimeofday(&t2, NULL);
        print_time("\tsolution", &t1, &t2);
#ifdef WITH_PAPI
        if (papi_supported) {
            if ((papi_err = PAPI_stop_counters(values, 7)) != PAPI_OK) {
                fprintf(stderr, "Could not get values: %s\n", PAPI_strerror(papi_err));
            }
            // PAPI_FP_OPS
            // PAPI_TOT_INS
            // PAPI_BR_INS
            // PAPI_LD_INS
            // PAPI_SR_INS
            // PAPI_L1_DCM
            // PAPI_L2_TCM
            printf("Performance counters: \n");
            printf("\tFP OPS: %ld\n", values[0]);
            printf("\tTOT INS: %ld\n", values[1]);
            printf("\tBR INS: %ld\n", values[2]);
            printf("\tLD INS: %ld\n", values[3]);
            printf("\tSR INS: %ld\n", values[4]);
            printf("\tL1 DCM: %ld\n", values[5]);
            printf("\tL2 TCM: %ld\n", values[6]);
        }
#endif
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
    } else if (scheduler == SEQ) {
        gettimeofday(&t1, NULL);
        seqAllocation(m->getRootNode());
        gettimeofday(&t2, NULL);
        print_time("\tallocation", &t1, &t2);

#ifdef WITH_PAPI
        if (papi_supported) {
            if ((papi_err = PAPI_start_counters(events, 7)) != PAPI_OK) {
                fprintf(stderr, "Could not start counters: %s\n", PAPI_strerror(papi_err));
            }
        }
#endif
        gettimeofday(&t1, NULL);
        seqElimination(m->getRootNode());
        gettimeofday(&t2, NULL);
        print_time("\tfactorization", &t1, &t2);

        gettimeofday(&t1, NULL);
        seqBackwardSubstitution(m->getRootNode());
        gettimeofday(&t2, NULL);
        print_time("\tsolution", &t1, &t2);
#ifdef WITH_PAPI
        if (papi_supported) {
            if ((papi_err = PAPI_stop_counters(values, 7)) != PAPI_OK) {
                fprintf(stderr, "Could not get values: %s\n", PAPI_strerror(papi_err));
            }
            // PAPI_FP_OPS
            // PAPI_TOT_INS
            // PAPI_BR_INS
            // PAPI_LD_INS
            // PAPI_SR_INS
            // PAPI_L1_DCM
            // PAPI_L2_TCM
            printf("Performance counters: \n");
            printf("\tFP OPS: %ld\n", values[0]);
            printf("\tTOT INS: %ld\n", values[1]);
            printf("\tBR INS: %ld\n", values[2]);
            printf("\tLD INS: %ld\n", values[3]);
            printf("\tSR INS: %ld\n", values[4]);
            printf("\tL1 DCM: %ld\n", values[5]);
            printf("\tL2 TCM: %ld\n", values[6]);

        }
#endif
    }

    delete m;
    //delete lib;

    return 0;
}
