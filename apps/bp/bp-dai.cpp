#include "Galois/Galois.h"
#include "Galois/Statistic.h"
#include "Galois/Graphs/Graph.h"
#include "Lonestar/Banner.h"
#include "Lonestar/CommandLine.h"

#include "Exp/PriorityScheduling/WorkListTL.h"

#include <dai/daialg.h>
#include <dai/alldai.h>
#include <strstream>
#include <dai/bp.h>



static const char* name = "Belief propagation";
static const char* description = "Belief propagation on Ising Grids";
static const char* url = 0;
static const char* help = "[-algo N] <N> <hardness> <seed> <max iterations> <damping>";

static const double TOL = 1e-4;
static int MaxIterations;

long GlobalTime = 0;

double nextRand() {
  return rand() / (double) RAND_MAX;
}

//! Generate random Ising grid
//!  N*N discrete variables, X_i, \phi(X_i) in {0, 1} (spin)
//!  \phi_ij(X_i, X_j) = e^{\lambda*C} if x_i = x_j or e^{-\lambda*C} otherwise
//!  \lambda in [-0.5, 0.5]
void generateInput(int N, double hardness, int seed,
    std::vector<dai::Var>& variables, std::vector<dai::Factor>& factors) {
  srand(seed);
  variables.clear();
  for (int i = 0; i < N*N; ++i)
    variables.push_back(dai::Var(i, 2));
  
  factors.clear();
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      if (i >= 1) {
        factors.push_back(createFactorIsing(variables[i*N+j],
              variables[(i-1)*N+j],
              (nextRand() - 0.5) * hardness));
      }
      if (j >= 1) {
        factors.push_back(createFactorIsing(variables[i*N+j],
              variables[i*N+(j-1)],
              (nextRand() - 0.5) * hardness));
      }
      factors.push_back(createFactorIsing(variables[i*N+j], nextRand()));
    }
  }
}

int main(int argc,  const char **argv) {
  int algo = 0;
  std::vector<const char*> args = parse_command_line(argc, argv, help);
  Exp::parse_worklist_command_line(args);

  for (std::vector<const char*>::iterator ii = args.begin(), ei = args.end(); ii != ei; ++ii) {
    if (strcmp(*ii, "-algo") == 0 && ii + 1 != ei) {
      algo = atoi(ii[1]);
      ii = args.erase(ii);
      ii = args.erase(ii);
      --ii;
      ei = args.end();
    }
  }
  
  if (args.size() < 5) {
    std::cerr << "incorrect number of arguments, use -help for usage information\n";
    return 1;
  }

  int N = atoi(args[0]);
  double hardness = atof(args[1]);
  int seed = atoi(args[2]);
  MaxIterations = atoi(args[3]);
  double damping = atof(args[4]);

  printBanner(std::cout, name, description, url);
  std::cout << "N: " << N << " hardness: " << hardness << " seed: " << seed
    << " maxiterations: " << MaxIterations << " damping: " << damping << " algo: " << algo << "\n";

  std::vector<dai::Var> variables;
  std::vector<dai::Factor> factors;
  generateInput(N, hardness, seed, variables, factors);
  dai::FactorGraph fg(factors);

  std::string algostring;
  switch (algo) {
    case 5: algostring = "SEQPRIASYNC"; break;
    case 4: algostring = "SEQPRI"; break;
    case 3: algostring = "PARALL"; break;
    case 2: algostring = "SEQMAX"; break;
    case 1: algostring = "SEQFIX"; break;
    case 0:
    default: algostring = "SEQRND"; break;
  }

  dai::PropertySet opts;
  size_t maxiter = MaxIterations;
  size_t verb = 3;
  opts.set("maxiter", maxiter);  // Maximum number of iterations
  opts.set("tol", TOL);          // Tolerance for convergence
  opts.set("verbose", verb);     // Verbosity (amount of output generated)
  opts.set("damping", damping);  //
  opts.set("worklist", Exp::WorklistName);
  opts.set("logdomain", false);

  dai::BP bp(fg, opts("updates", algostring));

  bp.init();

  Galois::StatTimer T("bp");
  T.start();
  bp.run();
  T.stop();
  std::cout << "Time: " << GlobalTime << "\n";

  if (!skipVerify) {
    Galois::StatTimer verify("verify");
    verify.start();
    std::cout << "Starting verification...\n";
    verb = 0;
    dai::BP verifier(fg, opts("updates", std::string("SEQMAX"))("verbose", verb));
    verifier.init();
    verifier.run();
    
    dai::Real r = 0;
    for (size_t i = 0; i < fg.nrVars(); ++i) {
      r += dai::dist(bp.belief(fg.var(i)), verifier.belief(fg.var(i)), dai::DISTKL);
    }
    std::cout << "Average KL distance from SEQMAX: " << r / (dai::Real) fg.nrVars() << "\n";
    verify.stop();
  }
#if 0
  Galois::StatTimer Texact("exact");
  dai::ExactInf ei(fg, dai::PropertySet()("verbose",(size_t)0) );
  ei.init();
  Texact.start();
  ei.run();
  Texact.stop();
#endif
 
  return 0;
}



