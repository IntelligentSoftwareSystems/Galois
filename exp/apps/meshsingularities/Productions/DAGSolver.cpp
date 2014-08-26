#include "DynamicLib.h"
#include "ExternalMesh/Analysis.hpp"
#include "ExternalMesh/Node.hpp"
#include "ExternalMesh/Element.hpp"
#include "ExternalMesh/Mesh.hpp"

#include "EquationSystem.h"

#include "Galois/Galois.h"
#include "Galois/Accumulator.h"
#include "Galois/Bag.h"
#include "Galois/Statistic.h"
#include "Galois/UnionFind.h"
#include "Galois/Graph/LCGraph.h"
#include "Galois/ParallelSTL/ParallelSTL.h"
#include "Galois/Graph/LC_Morph_Graph.h"

#include "Lonestar/BoilerPlate.h"

#include <cilk/cilk.h>

#include "llvm/Support/CommandLine.h"

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

void cilk_alloc_tree(Node *n, DynamicLib *lib)
{
    n->allocateSystem();
    if (n->getRight() != NULL && n->getLeft() != NULL) {
        cilk_spawn cilk_alloc_tree(n->getLeft(), lib);
        cilk_spawn cilk_alloc_tree(n->getRight(), lib);
        cilk_sync;
        n->mergeProduction = (void(*)(double**, double*, double**, double*, double**, double*))
                lib->resolvSymbol(n->getProduction());
    } else {
        n->preprocessProduction = (void(*)(double**, double*, double**, double*))lib->resolvSymbol(n->getProduction());
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

int main(int argc, char ** argv)
{
    LonestarStart(argc, argv, name, desc, url);

    DynamicLib *lib = new DynamicLib(prodlib);
    lib->load();
    Mesh *m  = Mesh::loadFromFile(treefile.c_str());

    Analysis::enumerateDOF(m);
    Analysis::doAnalise(m);

    cilk_alloc_tree(m->getRootNode(), lib);
    cilk_do_elimination(m->getRootNode());
    cilk_do_backward_substitution(m->getRootNode());


    delete m;
    delete lib;

    return 0;
}
