#include "galois/Galois.h"
#include "galois/Timer.h"
#include "galois/Graph/Graph.h"
#include "llvm/Support/CommandLine.h"
#include "Lonestar/BoilerPlate.h"

#include "galois/PriorityScheduling.h"

#include <dai/daialg.h>
#include <dai/alldai.h>
#include <strstream>
#include <dai/bp.h>

namespace cll = llvm::cl;

static const char* name = "Belief propagation";
static const char* desc = "Performs belief propagation on Ising Grids";
static const char* url  = 0;

static cll::opt<int> algo("algo", cll::desc("Node to start search from"),
                          cll::init(1));
static cll::opt<int> N(cll::Positional, cll::desc("<N>"), cll::Required);
static cll::opt<double> hardness(cll::Positional, cll::desc("<hardness>"),
                                 cll::Required);
static cll::opt<int> seed(cll::Positional, cll::desc("<seed>"), cll::Required);
static cll::opt<int> MaxIterations(cll::Positional,
                                   cll::desc("<max iterations>"),
                                   cll::Required);
static cll::opt<double> damping(cll::Positional, cll::desc("<damping>"),
                                cll::Required);

static const double TOL = 1e-4;

long GlobalTime = 0;

double nextRand() { return rand() / (double)RAND_MAX; }

//! Generate random Ising grid
//!  N*N discrete variables, X_i, \phi(X_i) in {0, 1} (spin)
//!  \phi_ij(X_i, X_j) = e^{\lambda*C} if x_i = x_j or e^{-\lambda*C} otherwise
//!  \lambda in [-0.5, 0.5]
void generateInput(int N, double hardness, int seed,
                   std::vector<dai::Var>& variables,
                   std::vector<dai::Factor>& factors) {
  srand(seed);
  variables.clear();
  for (int i = 0; i < N * N; ++i)
    variables.push_back(dai::Var(i, 2));

  factors.clear();
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      if (i >= 1) {
        factors.push_back(createFactorIsing(variables[i * N + j],
                                            variables[(i - 1) * N + j],
                                            (nextRand() - 0.5) * hardness));
      }
      if (j >= 1) {
        factors.push_back(createFactorIsing(variables[i * N + j],
                                            variables[i * N + (j - 1)],
                                            (nextRand() - 0.5) * hardness));
      }
      factors.push_back(createFactorIsing(variables[i * N + j], nextRand()));
    }
  }
}

int main(int argc, char** argv) {
  LonestarStart(argc, argv, name, desc, url);

  std::cout << "N: " << N << " hardness: " << hardness << " seed: " << seed
            << " maxiterations: " << MaxIterations << " damping: " << damping
            << " algo: " << algo << "\n";

  std::vector<dai::Var> variables;
  std::vector<dai::Factor> factors;
  generateInput(N, hardness, seed, variables, factors);
  dai::FactorGraph fg(factors);

  std::string algostring;
  switch (algo) {
  case 5:
    algostring = "SEQPRIASYNC";
    break;
  case 4:
    algostring = "SEQPRI";
    break;
  case 3:
    algostring = "PARALL";
    break;
  case 2:
    algostring = "SEQMAX";
    break;
  case 1:
    algostring = "SEQFIX";
    break;
  case 0:
  default:
    algostring = "SEQRND";
    break;
  }

  // Put values in the right types for propertyset
  dai::PropertySet opts;
  size_t maxiter = MaxIterations;
  size_t verb    = 3;
  std::string worklist(Exp::WorklistName);
  std::ostringstream dampingStream;
  dampingStream << damping;

  opts.set("maxiter", maxiter); // Maximum number of iterations
  opts.set("tol", TOL);         // Tolerance for convergence
  opts.set("verbose", verb);    // Verbosity (amount of output generated)
  opts.set("damping", dampingStream.str());
  opts.set("worklist", worklist);
  opts.set("logdomain", false);

  dai::BP bp(fg, opts("updates", algostring));

  bp.init();

  galois::StatTimer T;
  T.start();
  bp.run();
  T.stop();
  std::cout << "Time: " << GlobalTime << "\n";

  if (!skipVerify) {
    galois::StatTimer verify("verify");
    verify.start();
    std::cout << "Starting verification...\n";
    verb = 0;
    dai::BP verifier(fg,
                     opts("updates", std::string("SEQMAX"))("verbose", verb));
    verifier.init();
    verifier.run();

    dai::Real r = 0;
    for (size_t i = 0; i < fg.nrVars(); ++i) {
      r += dai::dist(bp.belief(fg.var(i)), verifier.belief(fg.var(i)),
                     dai::DISTKL);
    }
    std::cout << "Average KL distance from SEQMAX: "
              << r / (dai::Real)fg.nrVars() << "\n";
    verify.stop();
  }
#if 0
  galois::StatTimer Texact("exact");
  dai::ExactInf ei(fg, dai::PropertySet()("verbose",(size_t)0) );
  ei.init();
  Texact.start();
  ei.run();
  Texact.stop();
#endif

  return 0;
}
