#pragma once

enum DataCommMode { noData, bitsetData, offsetsData, onlyData, dataSplitFirst, dataSplit };

extern DataCommMode enforce_data_mode;

