#ifndef _GNN_H_
#define _GNN_H_

#include "galois/Galois.h"
#include "galois/Reduction.h"
#include "galois/Timer.h"
#include "galois/ParallelSTL.h"
#include "llvm/Support/CommandLine.h"
#include "Lonestar/BoilerPlate.h"
#include "galois/runtime/Profile.h"
#include <boost/iterator/transform_iterator.hpp>

namespace cll = llvm::cl;
static cll::opt<std::string> dataset(cll::Positional, cll::desc("<dataset name>"), cll::Required); // 'cora', 'citeseer', 'pubmed'
static cll::opt<std::string> model("m", cll::desc("Model string"), cll::init("gcn")); // 'gcn', 'gcn_cheby', 'dense'
static cll::opt<float> learning_rate("lr", cll::desc("Initial learning rate (default value 0.01)"), cll::init(0.01));
static cll::opt<unsigned> epochs("k", cll::desc("number of epoch, i.e. iterations (default value 1)"), cll::init(1));
static cll::opt<unsigned> hidden1("h", cll::desc("Number of units in hidden layer 1 (default value 16)"), cll::init(16));
static cll::opt<float> dropout_rate("d", cll::desc("Dropout rate (1 - keep probability) (default value 0.5)"), cll::init(0.5));
static cll::opt<float> weight_decay("wd", cll::desc("Weight for L2 loss on embedding matrix (default value 5e-4)"), cll::init(5e-4));
static cll::opt<float> early_stopping("es", cll::desc("Tolerance for early stopping (# of epochs) (default value 10)"), cll::init(10));
static cll::opt<unsigned> max_degree("md", cll::desc("Maximum Chebyshev polynomial degree (default value 3)"), cll::init(3));
#define CHUNK_SIZE 256

#endif
