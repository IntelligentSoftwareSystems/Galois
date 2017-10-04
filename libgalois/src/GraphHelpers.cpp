/** Graph helper functions -*- C++ -*-
 * @file GraphHelpers.h
 * @section License
 *
 * This file is part of Galois.  Galois is a framework to exploit
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
 * Copyright (C) 2017, The University of Texas at Austin. All rights
 * reserved.
 *
 * @section Description
 *
 * Contains functions that can be done on various graphs with a particular
 * interface.
 *
 * @author Loc Hoang <l_hoang@utexas.edu>
 */
#include <galois/graphs/GraphHelpers.h>

namespace galois {
namespace graphs {
namespace internal {

uint32_t determine_block_division(uint32_t numDivisions,
                                         std::vector<unsigned>& scaleFactor) {
  uint32_t numBlocks = 0;

  if (scaleFactor.empty()) {
    // if scale factor isn't specified, everyone gets the same amount
    numBlocks = numDivisions;

    // scale factor holds a prefix sum of the scale factor
    for (uint32_t i = 0; i < numDivisions; i++) {
      scaleFactor.push_back(i + 1);
    }
  } else {
    assert(scaleFactor.size() == numDivisions);
    assert(numDivisions >= 1);

    // get numDivisions number of blocks we need + save a prefix sum of the scale
    // factor vector to scaleFactor
    for (uint32_t i = 0; i < numDivisions; i++) {
      numBlocks += scaleFactor[i];
      scaleFactor[i] = numBlocks;
    }
  }

  return numBlocks;
}

}
}
}
