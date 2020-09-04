#ifndef GALOIS_DISTBENCH_OUTPUT_H
#define GALOIS_DISTBENCH_OUTPUT_H

#include <string>
#include <fstream>
#include "galois/gIO.h"

std::string makeOutputFilename(const std::string& outputDir);

template <typename T>
void writeOutput(const std::string& outputDir, const std::string& /*fieldName*/,
                 T* values, size_t length, uint64_t* IDs) {
  std::string filename = makeOutputFilename(outputDir);

  std::ofstream outputFile(filename.c_str());

  for (size_t i = 0; i < length; i++) {
    outputFile << *(IDs++) << " " << *(values++) << "\n";
  }

  galois::gPrint("Output written to: ", filename, "\n");
}

#endif
