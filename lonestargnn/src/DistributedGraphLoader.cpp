/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
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
        clEnumValN(HOVC, "hovc", "Outgoing Hybrid Vertex-Cut"),
        clEnumValN(HIVC, "hivc", "Incoming Hybrid Vertex-Cut"),
        clEnumValN(CART_VCUT, "cvc", "Cartesian Vertex-Cut of oec"),
        clEnumValN(CART_VCUT_IEC, "cvc-iec", "Cartesian Vertex-Cut of iec"),
        //clEnumValN(CEC, "cec", "Custom edge cut from vertexID mapping"),
        clEnumValN(GINGER_O, "ginger-o", "ginger, outgiong edges, using CuSP"),
        clEnumValN(GINGER_I, "ginger-i", "ginger, incoming edges, using CuSP"),
        clEnumValN(FENNEL_O, "fennel-o", "fennel, outgoing edge cut, using CuSP"),
        clEnumValN(FENNEL_I, "fennel-i", "fennel, incoming edge cut, using CuSP"),
        clEnumValN(SUGAR_O, "sugar-o", "fennel, incoming edge cut, using CuSP"),
        clEnumValEnd),
    cll::init(OEC));
