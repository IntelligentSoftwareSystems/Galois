#pragma once
enum DataCommMode { noData, bitsetData, offsetsData, onlyData, dataSplitFirst, 
                    dataSplit };

extern DataCommMode enforce_data_mode;

template<typename DataType>
DataCommMode get_data_mode(size_t num_selected, size_t num_total) {
  DataCommMode data_mode = noData;
  if (enforce_data_mode != noData) {
    data_mode = enforce_data_mode;
  } else { // == noData; set appropriately
    if (num_selected == 0) {
      data_mode = noData;
    } else {
      size_t bitset_alloc_size = ((num_total + 63)/64) * sizeof(uint64_t) + (2 * sizeof(size_t));

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
