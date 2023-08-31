/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2019, The University of Texas at Austin. All rights reserved.
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

/**
 * @file DistributedGraphLoader.cpp
 *
 * Contains definitions for command line arguments related to distributed
 * graph loading.
 */

#include "DistributedGraphLoader.h"

using namespace galois::graphs;

namespace cll = llvm::cl;

cll::opt<PARTITIONING_SCHEME> partitionScheme(
    "partition", cll::desc("Type of partitioning."),
    cll::values(
        clEnumValN(OEC, "oec", "Outgoing Edge-Cut (default)"),
        clEnumValN(IEC, "iec", "Incoming Edge-Cut"),
        clEnumValN(CART_VCUT, "cvc", "Cartesian Vertex-Cut of oec"),
        clEnumValN(CART_VCUT_IEC, "cvc-iec", "Cartesian Vertex-Cut of iec"),
        clEnumValN(GNN_OEC, "g-oec", "gnn oec: train nodes evenly distributed"),
        clEnumValN(GNN_CVC, "g-cvc",
                   "gnn cvc: train nodes evenly distributed")),
    cll::init(GNN_OEC));

cll::opt<bool> useWMD("useWMD", cll::desc("true if the input graph is"
                                          " SHAD WMD graph format."
                                          " Otheriwse, set false."),
                       cll::init(false));
