#ifndef ASC_READER_H
#define ASC_READER_H

#include <stddef.h>
#include <stdio.h>
#include <string>
#include "../model/Map.h"

class AsciiReader {
public:
  Map* read(const std::string filename);

private:
  static int readLine(FILE* f, char* buffer, const size_t buffersize,
                      size_t* line_number);

  static Map* convert(double** coords, const size_t nRows, const size_t nCols);
};

#endif // ASC_READER_H
