/*
 */

/**
 * @file DataCommMode.h
 *
 * Contains the DataCommMode enumeration and a function that chooses a data
 * comm mode based on its arguments.
 */
#pragma once

//! Enumeration of data communication modes that can be used in sychronization
//! @todo document the enums in doxygen
enum DataCommMode { 
  noData, //!< send no data
  bitsetData
  offsetsData,
  gidsData,
  onlyData,
  dataSplitFirst,
  dataSplit
};

//! If this is set, then always used the data mode it is set to
extern DataCommMode enforce_data_mode;

/**
 * Given a size of a subset of elements to send and the total number of elements,
 * determine an appropriate data mode to use for sending out the data during
 * synchronization.
 *
 * @tparam DataType type of the data to be synchronized
 *
 * @param num_selected number of elements to send out (subset of num_total)
 * @param num_total total number of elements that exist
 *
 * @returns an appropriate DataCommMode to use for synchronization
 */
template<typename DataType>
DataCommMode get_data_mode(size_t num_selected, size_t num_total) {
  DataCommMode data_mode = noData;
  if (enforce_data_mode != noData) {
    data_mode = enforce_data_mode;
  } else { // no enforced mode, so find an appropriate mode
    if (num_selected == 0) {
      data_mode = noData;
    } else {
      size_t bitset_alloc_size = ((num_total + 63)/64) * sizeof(uint64_t) +
                                 (2 * sizeof(size_t));

      size_t onlyDataSize = (num_total * sizeof(DataType));
      size_t bitsetDataSize = (num_selected * sizeof(DataType)) +
                  bitset_alloc_size +
                  sizeof(num_selected);
      size_t offsetsDataSize = (num_selected * sizeof(DataType)) +
                  (num_selected * sizeof(unsigned int)) + sizeof(size_t) +
                  sizeof(num_selected);
      // find the minimum size one
      if (bitsetDataSize < offsetsDataSize) {
        if (bitsetDataSize < onlyDataSize) {
          data_mode = bitsetData;
        } else {
          data_mode = onlyData;
        }
      } else {
        if (offsetsDataSize < onlyDataSize) {
          data_mode = offsetsData;
        } else {
          data_mode = onlyData;
        }
      }
    }
  }
  return data_mode;
}
