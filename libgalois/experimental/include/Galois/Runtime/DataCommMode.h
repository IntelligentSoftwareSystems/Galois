#pragma once
// TODO: move to libdist?

enum DataCommMode { noData, offsetsData, bitsetData, onlyData, dataSplitFirst, dataSplit };

extern DataCommMode enforce_data_mode;

