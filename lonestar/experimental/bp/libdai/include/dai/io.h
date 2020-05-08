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

/*  This file is part of libDAI - http://www.libdai.org/
 *
 *  Copyright (c) 2006-2011, The libDAI authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license that can be found
 * in the LICENSE file.
 */

/// \file
/// \brief Provides functionality for input/output of data structures in various
/// file formats

#ifndef __defined_libdai_io_h
#define __defined_libdai_io_h

#include <dai/factor.h>
#include <vector>
#include <map>

namespace dai {

/// Reads factor graph (as a pair of a variable vector and factor vector) from a
/// file in the UAI approximate inference challenge format
/** \param[in] filename The filename (usually ends with ".uai")
 *  \param[in] verbose The amount of output sent to cout
 *  \param[out] vars Array of variables read from the file
 *  \param[out] factors Array of factors read from the file
 *  \param[out] permutations Array of permutations, which permute between libDAI
 * canonical ordering and ordering specified by the file \see
 * http://www.cs.huji.ac.il/project/UAI10 and http://graphmod.ics.uci.edu/uai08
 */
void ReadUaiAieFactorGraphFile(const char* filename, size_t verbose,
                               std::vector<Var>& vars,
                               std::vector<Factor>& factors,
                               std::vector<Permute>& permutations);

/// Reads evidence (a mapping from observed variable labels to the observed
/// values) from a file in the UAI approximate inference challenge format
/** \param[in] filename The filename (usually ends with ".uai.evid")
 *  \param[in] verbose The amount of output sent to cout
 *  \see http://www.cs.huji.ac.il/project/UAI10 and
 * http://graphmod.ics.uci.edu/uai08
 */
std::vector<std::map<size_t, size_t>>
ReadUaiAieEvidenceFile(const char* filename, size_t verbose);

} // end of namespace dai

#endif
