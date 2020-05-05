/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 */

#include "llvm/Support/CommandLine.h"
/**
 * Common commandline parameters to for matrix completion algorithms
 */
namespace cll = llvm::cl;
static cll::opt<std::string>
    inputFilename(cll::Positional, cll::desc("<input file>"), cll::Required);

// (Purdue, Neflix): 0.012, (Purdue, Yahoo Music): 0.00075, (Purdue, HugeWiki):
// 0.001 Intel: 0.001 Bottou: 0.1
static cll::opt<float> learningRate("learningRate",
                                    cll::desc("learning rate parameter [alpha] "
                                              "for Bold, Bottou, Intel and "
                                              "Purdue step size function"),
                                    cll::init(0.012));

// (Purdue, Netflix): 0.015, (Purdue, Yahoo Music): 0.01,
// (Purdue, HugeWiki): 0.0, Intel: 0.9
static cll::opt<float> decayRate("decayRate",
                                 cll::desc("decay rate parameter [beta] for "
                                           "Intel and Purdue step size "
                                           "function"),
                                 cll::init(0.015));
// (Purdue, Netflix): 0.05, (Purdue, Yahoo Music): 1.0, (Purdue, HugeWiki): 0.01
// Intel: 0.001
static cll::opt<float> lambda("lambda",
                              cll::desc("regularization parameter [lambda]"),
                              cll::init(0.05));

static cll::opt<unsigned> usersPerBlock("usersPerBlock",
                                        cll::desc("users per block"),
                                        cll::init(2048));
static cll::opt<unsigned> itemsPerBlock("itemsPerBlock",
                                        cll::desc("items per block"),
                                        cll::init(350));
static cll::opt<float>
    tolerance("tolerance", cll::desc("convergence tolerance"), cll::init(0.01));

static cll::opt<bool> useSameLatentVector("useSameLatentVector",
                                          cll::desc("initialize all nodes to "
                                                    "use same latent vector"),
                                          cll::init(false));

// Regarding algorithm termination
static cll::opt<unsigned> maxUpdates("maxUpdates",
                                     cll::desc("Max number of times to update "
                                               "latent vectors (default 100)"),
                                     cll::init(100));
