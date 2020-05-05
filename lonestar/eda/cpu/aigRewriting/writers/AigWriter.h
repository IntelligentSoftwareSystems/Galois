/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
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

#ifndef AIGWRITER_H_
#define AIGWRITER_H_

#include <fstream>
#include <iostream>
#include <string>

#include "../subjectgraph/aig/Aig.h"
#include "galois/Galois.h"

typedef aig::Aig Aig;

class AigWriter {

private:
  std::ofstream aigerFile;
  std::string path;

  void writeAagHeader(Aig& aig);
  void writeLatchesAag(Aig& aig);
  void writeAndsAag(Aig& aig);

  void writeAigHeader(Aig& aig);
  void writeLatchesAig(Aig& aig);
  void writeAndsAig(Aig& aig);

  void writeInputs(Aig& aig);
  void writeOutputs(Aig& aig);
  void writeSymbolTable(Aig& aig);

  void encode(unsigned x);

public:
  AigWriter();
  AigWriter(std::string path);
  virtual ~AigWriter();

  void setFile(std::string path);
  bool isOpen();
  void close();

  void writeAag(Aig& aig);
  void writeAig(Aig& aig);
};

#endif /* AIGWRITER_H_ */
