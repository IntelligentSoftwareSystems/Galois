#include "Galois/Galois.h"
#include "Galois/Statistic.h"
#include "Galois/Graphs/Graph.h"
#include "Lonestar/Banner.h"
#include "Lonestar/CommandLine.h"

#include <dai/daialg.h>
#include <dai/alldai.h>
#include <strstream>
#include <dai/bp.h>

static const char* name = "Belief propagation";
static const char* description = "Belief propagation on Ising Grids";
static const char* url = 0;
static const char* help = "[-algo N] <N> <hardness> <seed> <max iterations> <damping>";

//static const double DAMPING = 0.2;
static const double TOL = 1e-10;
static int MaxIterations;

double nextRand() {
  return rand() / (double) RAND_MAX;
}

int main(int argc,  const char **argv) {
  int algo = 0;
  std::vector<const char*> args = parse_command_line(argc, argv, help);
  for (std::vector<const char*>::iterator ii = args.begin(), ei = args.end(); ii != ei; ++ii) {
    if (strcmp(*ii, "-algo") == 0 && ii + 1 != ei) {
      algo = atoi(ii[1]);
      ii = args.erase(ii);
      ii = args.erase(ii);
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

  //! Generate random Ising grid
  //!  N*N discrete variables, X_i, \phi(X_i) in {0, 1} (spin)
  //!  \phi_ij(X_i, X_j) = e^{\lambda*C} if x_i = x_j or e^{-\lambda*C} otherwise
  //!  \lambda in [-0.5, 0.5]
  srand(seed);
  std::vector<dai::Var> variables;
  for (int i = 0; i < N*N; ++i)
    variables.push_back(dai::Var(i, 2));
  
  std::vector<dai::Factor> factors;
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

  dai::FactorGraph fg(factors);

  dai::PropertySet opts;
  size_t maxiter = MaxIterations;
  size_t verb = 3;
  opts.set("maxiter", maxiter);  // Maximum number of iterations
  opts.set("tol", TOL);          // Tolerance for convergence
  opts.set("verbose", verb);     // Verbosity (amount of output generated)
  opts.set("damping", damping);  //

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
  dai::BP bp(fg, opts("updates", algostring)("logdomain",false));

  bp.init();

  Galois::StatTimer T("bp");
  T.start();
  bp.run();
  T.stop();

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



