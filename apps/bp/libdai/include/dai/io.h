/*  This file is part of libDAI - http://www.libdai.org/
 *
 *  Copyright (c) 2006-2011, The libDAI authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
 */


/// \file
/// \brief Provides functionality for input/output of data structures in various file formats


#ifndef __defined_libdai_io_h
#define __defined_libdai_io_h


#include <dai/factor.h>
#include <vector>
#include <map>


namespace dai {


/// Reads factor graph (as a pair of a variable vector and factor vector) from a file in the UAI approximate inference challenge format
/** \param[in] filename The filename (usually ends with ".uai")
 *  \param[in] verbose The amount of output sent to cout
 *  \param[out] vars Array of variables read from the file
 *  \param[out] factors Array of factors read from the file
 *  \param[out] permutations Array of permutations, which permute between libDAI canonical ordering and ordering specified by the file
 *  \see http://www.cs.huji.ac.il/project/UAI10 and http://graphmod.ics.uci.edu/uai08
 */
void ReadUaiAieFactorGraphFile( const char *filename, size_t verbose, std::vector<Var>& vars, std::vector<Factor>& factors, std::vector<Permute>& permutations );


/// Reads evidence (a mapping from observed variable labels to the observed values) from a file in the UAI approximate inference challenge format
/** \param[in] filename The filename (usually ends with ".uai.evid")
 *  \param[in] verbose The amount of output sent to cout
 *  \see http://www.cs.huji.ac.il/project/UAI10 and http://graphmod.ics.uci.edu/uai08
 */
std::vector<std::map<size_t, size_t> > ReadUaiAieEvidenceFile( const char* filename, size_t verbose );


} // end of namespace dai


#endif
