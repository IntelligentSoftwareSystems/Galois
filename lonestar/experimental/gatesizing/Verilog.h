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

#include <string>
#include <unordered_map>
#include <unordered_set>

#include "CellLib.h"

#ifndef GALOIS_VERILOG_H
#define GALOIS_VERILOG_H

struct VerilogGate;
struct VerilogWire;

struct VerilogPin {
  std::string name;
  VerilogGate* gate;
  VerilogWire* wire;
};

struct VerilogWire {
  std::string name;
  VerilogPin* root;
  WireLoad* wireLoad;
  std::unordered_set<VerilogPin*> leaves;
};

struct VerilogGate {
  std::string name;
  Cell* cell;
  std::unordered_set<VerilogPin*> inPins, outPins;
};

struct VerilogModule {
  std::string name;
  std::unordered_map<std::string, VerilogPin*> inputs, outputs;
  std::unordered_map<std::string, VerilogGate*> gates;
  std::unordered_map<std::string, VerilogWire*> wires;
  CellLib* cellLib;

  VerilogModule();
  ~VerilogModule();

  void read(std::string inName, CellLib* lib);
  void clear();
  void printDebug();
  void write(std::string outName);
};

#endif // GALOIS_VERILOG_H
