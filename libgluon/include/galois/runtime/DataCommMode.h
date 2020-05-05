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

/*
 */

/**
 * @file DataCommMode.h
 *
 * Contains the DataCommMode enumeration and a function that chooses a data
 * comm mode based on its arguments.
 */
#pragma once

//! Enumeration of data communication modes that can be used in synchronization
//! @todo document the enums in doxygen
enum DataCommMode {
  noData, //!< send no data
  bitsetData,
  offsetsData,
  gidsData,
  onlyData,
  dataSplitFirst, // NOT USED
  dataSplit       // NOT USED
};

//! If some mode is to be enforced, set this variable
//! @todo using a global is not great, but current problem is that GPU code
//! assumes variable and would take some reorg to fix
extern DataCommMode enforcedDataMode;

/**
 * Given a size of a subset of elements to send and the total number of
 * elements, determine an appropriate data mode to use for sending out the data
 * during synchronization.
 *
 * @tparam DataType type of the data to be synchronized
 *
 * @param num_selected number of elements to send out (subset of num_total)
 * @param num_total total number of elements that exist
 *
 * @returns an appropriate DataCommMode to use for synchronization
 */
template <typename DataType>
DataCommMode get_data_mode(size_t num_selected, size_t num_total) {
  DataCommMode data_mode = noData;
  if (enforcedDataMode != noData) {
    data_mode = enforcedDataMode;
  } else { // no enforced mode, so find an appropriate mode
    if (num_selected == 0) {
      data_mode = noData;
    } else if (num_selected == num_total) {
      data_mode = onlyData;
    } else {
      size_t bitset_alloc_size =
          ((num_total + 63) / 64) * sizeof(uint64_t) + (2 * sizeof(size_t));

      size_t bitsetDataSize = (num_selected * sizeof(DataType)) +
                              bitset_alloc_size + sizeof(num_selected);
      size_t offsetsDataSize = (num_selected * sizeof(DataType)) +
                               (num_selected * sizeof(unsigned int)) +
                               sizeof(size_t) + sizeof(num_selected);
      // find the minimum size one
      if (bitsetDataSize < offsetsDataSize) {
        data_mode = bitsetData;
      } else {
        data_mode = offsetsData;
      }
    }
  }
  return data_mode;
}
