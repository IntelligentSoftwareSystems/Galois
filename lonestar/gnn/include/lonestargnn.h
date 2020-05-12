#pragma once

#include "llvm/Support/CommandLine.h"

namespace cll = llvm::cl;
static cll::opt<std::string> dataset(cll::Positional, 
    cll::desc("<dataset name>"), cll::Required); // 'cora', 'citeseer', 'pubmed'
//static cll::opt<std::string> model("m", 
//  cll::desc("Model string"), cll::init("gcn")); // 'gcn', 'gcn_cheby', 'dense'
static cll::opt<int> epochs("k",
    cll::desc("number of epoch, i.e. iterations (default value 1)"), cll::init(1));
static cll::opt<unsigned> num_conv_layers("nc",
    cll::desc("number of convolutional layers, (default value 2)"), cll::init(2));
static cll::opt<unsigned> hidden1("h",
    cll::desc("Number of units in hidden layer 1 (default value 16)"), cll::init(16));
static cll::opt<float> learning_rate("lr", 
    cll::desc("Initial learning rate (default value 0.01)"), cll::init(0.01));
static cll::opt<float> dropout_rate("dr", 
    cll::desc("Dropout rate (1 - keep probability) (default value 0.5)"), cll::init(0.5));
static cll::opt<float> weight_decay("wd",
    cll::desc("Weight for L2 loss on embedding matrix (default value 5e-4)"), cll::init(5e-4));
static cll::opt<float> early_stopping("es",
    cll::desc("Tolerance for early stopping (# of epochs) (default value 10)"), cll::init(10));
static cll::opt<bool> is_single_class("sc", 
    cll::desc("single-class or multi-class label (default single)"), cll::init(1));
static cll::opt<bool> do_validate("dv", cll::desc("enable validation"), cll::init(1));
static cll::opt<bool> do_test("dt", cll::desc("enable test"), cll::init(1));
static cll::opt<bool> add_selfloop("sl", cll::desc("add selfloop"), cll::init(0));
static cll::opt<bool> add_l2norm("l2", cll::desc("add an l2_norm layer"), cll::init(0));
static cll::opt<bool> add_dense("d", cll::desc("add an dense layer"), cll::init(0));
static cll::opt<int> val_interval("vi", cll::desc("validation interval (default value 1)"), cll::init(1));
static cll::opt<unsigned> neighbor_sample_sz("ns", cll::desc("neighbor sampling size (default value 0)"), cll::init(0));
static cll::opt<unsigned> subgraph_sample_sz("ss", cll::desc("subgraph sampling size (default value 0)"), cll::init(0));

//! standard global options to the benchmarks
extern llvm::cl::opt<bool> skipVerify;
extern llvm::cl::opt<int> numThreads;
extern llvm::cl::opt<std::string> statFile;

//! standard global options to the benchmarks
llvm::cl::opt<bool> skipVerify("noverify",
    llvm::cl::desc("Skip verification step (default value false)"), llvm::cl::init(false));
llvm::cl::opt<int>numThreads("t", llvm::cl::desc("Number of threads (default value 1)"), llvm::cl::init(1));
llvm::cl::opt<std::string> statFile("statFile",
    llvm::cl::desc("ouput file to print stats to (default value empty)"), llvm::cl::init(""));

