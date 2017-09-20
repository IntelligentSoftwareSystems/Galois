/** Local Computation graphs -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galoisis a framework to exploit
 * amorphous data-parallelism in irregular programs.
 *
 * Galois is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, version 2.1 of the
 * License.
 *
 * Galois is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with Galois.  If not, see
 * <http://www.gnu.org/licenses/>.
 *
 * @section Copyright
 *
 * Copyright (C) 2015, The University of Texas at Austin. All rights
 * reserved.
 *
 * @section Description
 *
 * There are two main classes, galois::graphs::FileGraph and LC_XXX_Graph
 * (e.g., galois::graphs::LC_CSR_Graph). The former represents the pure
 * structure of a graph (i.e., whether an edge exists between two nodes) and
 * cannot be modified. The latter allows values to be stored on nodes and
 * edges, but the structure of the graph cannot be modified.
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 */

#ifndef GALOIS_GRAPH_LCGRAPH_H
#define GALOIS_GRAPH_LCGRAPH_H

#include "LC_CSR_Graph.h"
#include "LC_InlineEdge_Graph.h"
#include "LC_Linear_Graph.h"
#include "LC_Morph_Graph.h"
#include "LC_InOut_Graph.h"
#include "LC_Adaptor_Graph.h"
#include "Util.h"

#endif
