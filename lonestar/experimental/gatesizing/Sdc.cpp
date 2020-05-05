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

#include "Sdc.h"
#include "FileReader.h"

#include <string>
#include <iostream>
#include <limits>
#include <cmath>

static void setRequiredTime(float delay, SDC* sdc) {
  if (sdc->targetDelay > delay) {
    sdc->targetDelay = delay;
    auto& g          = sdc->graph->g;
    for (auto ie : g.in_edges(sdc->graph->dummySink)) {
      auto& data             = g.getData(g.getEdgeDst(ie));
      data.rise.requiredTime = delay;
      data.fall.requiredTime = delay;
    }
  }
}

static void createClock(FileReader& fRd, SDC* sdc) {
  fRd.nextToken(); // get "-period"
  float delay = std::stof(fRd.nextToken());
  fRd.nextToken(); // get "-name"
  fRd.nextToken(); // get the name for clock port, e.g. clk
  fRd.nextToken(); // get "["
  fRd.nextToken(); // get "get_ports"
  fRd.nextToken(); // get "{"
  fRd.nextToken(); // get "clock"
  fRd.nextToken(); // get "}"
  fRd.nextToken(); // get "]"

  setRequiredTime(delay, sdc);
}

static std::vector<GNode> getGNodesForPorts(FileReader& fRd, SDC* sdc) {
  auto& g = sdc->graph->g;
  std::vector<GNode> nodes;

  fRd.nextToken(); // get "["
  for (std::string token = fRd.nextToken(); token != "]";
       token             = fRd.nextToken()) {
    if ("all_inputs" == token) {
      // all primary inputs
      for (auto oe : g.edges(sdc->graph->dummySrc)) {
        nodes.push_back(g.getEdgeDst(oe));
      }
    } else if ("all_outputs" == token) {
      // all primary outputs
      for (auto ie : g.in_edges(sdc->graph->dummySink)) {
        nodes.push_back(g.getEdgeDst(ie));
      }
    } else if ("all_registers" == token) {
      // skip for now
    } else {
      VerilogModule* m = sdc->vModule;
      auto iter        = m->inputs.find(token);
      auto pin =
          (iter != m->inputs.end()) ? iter->second : m->outputs.at(token);
      nodes.push_back(sdc->graph->nodeMap.at(pin));
    }
  }

  return nodes;
}

static void setMaxDelay(FileReader& fRd, SDC* sdc) {
  float delay = std::stof(fRd.nextToken());

  fRd.nextToken(); // get "-from"
  getGNodesForPorts(fRd, sdc);

  fRd.nextToken(); // get "-to"
  getGNodesForPorts(fRd, sdc);

  // now handles only all_outputs and assumes arrival time for iputs = 0.0
  setRequiredTime(delay, sdc);
}

static float getPrimaryInputSlew(FileReader& fRd, SDC* sdc) {
  float slew = 0.0;

  std::string token = fRd.nextToken();
  if ("-input_transition_fall" == token) {
    slew = -std::stof(fRd.nextToken()); // negative to signal falling edge
  } else if ("-input_transition_rise" == token) {
    slew = std::stof(fRd.nextToken());
  } else {
    fRd.pushToken(token);
  }

  return slew;
}

static void setSlewByDrivingCell(std::vector<GNode>& nodes, Cell* drivingCell,
                                 std::string pinName, float slew, SDC* sdc) {
  if (0.0 == slew) {
    return;
  }

  auto outPin    = drivingCell->outPins.at(pinName);
  auto inPin     = drivingCell->inPins.begin()->second;
  auto inPinName = inPin->name;
  auto& g        = sdc->graph->g;

  // rising slew gives falling edge if the timing arc is negative unate
  auto& posDelayMap = (slew > 0.0) ? outPin->cellRise : outPin->cellFall;
  auto& posTransMap =
      (slew > 0.0) ? outPin->riseTransition : outPin->fallTransition;
  auto posDelayIter =
      posDelayMap.find({inPinName, TIMING_SENSE_POSITIVE_UNATE});
  bool isPos = (posDelayIter != posDelayMap.end());

  // falling edge gives rising slew if the timing arc is negative unate
  auto& negDelayMap = (slew > 0.0) ? outPin->cellFall : outPin->cellRise;
  auto& negTransMap =
      (slew > 0.0) ? outPin->fallTransition : outPin->riseTransition;
  auto negDelayIter =
      negDelayMap.find({inPinName, TIMING_SENSE_NEGATIVE_UNATE});
  bool isNeg = (negDelayIter != negDelayMap.end());

  // timing arc not handled
  if (!isPos && !isNeg) {
    return;
  }

  auto& transTables =
      (isPos) ? posTransMap.at({inPinName, TIMING_SENSE_POSITIVE_UNATE})
              : negTransMap.at({inPinName, TIMING_SENSE_NEGATIVE_UNATE});

  for (auto n : nodes) {
    auto& data     = g.getData(n);
    auto wire      = data.pin->wire;
    data.totalNetC = wire->wireLoad->wireCapacitance(wire->leaves.size());

    data.totalPinC = 0.0;
    for (auto oe : g.edges(n)) {
      auto& oData = g.getData(g.getEdgeDst(oe));
      auto pin    = oData.pin;
      if (pin->gate) {
        data.totalPinC +=
            pin->gate->cell->inPins.at(pin->name)->riseCapacitance;
      }
    }

    std::vector<float> vTotalC = {std::abs(slew),
                                  data.totalPinC + data.totalNetC};
    auto& info = (slew > 0.0) ? ((isPos) ? data.rise : data.fall)
                              : ((isPos) ? data.fall : data.rise);
    info.slew = extractMaxFromTableSet(transTables, vTotalC).first;
  }
}

static void setDrivingCell(FileReader& fRd, SDC* sdc) {
  fRd.nextToken(); // get "-lib_cell"
  Cell* drivingCell = sdc->cellLib->cells.at(fRd.nextToken());

  fRd.nextToken(); // get "-pin"
  std::string pinName = fRd.nextToken();

  auto nodes = getGNodesForPorts(fRd, sdc);

  float slew = getPrimaryInputSlew(fRd, sdc);
  setSlewByDrivingCell(nodes, drivingCell, pinName, slew, sdc);

  slew = getPrimaryInputSlew(fRd, sdc);
  setSlewByDrivingCell(nodes, drivingCell, pinName, slew, sdc);
}

static void setLoad(FileReader& fRd, SDC* sdc) {
  fRd.nextToken(); // get "-pin_load"
  float pinC = std::stof(fRd.nextToken());
  auto nodes = getGNodesForPorts(fRd, sdc);
  for (auto n : nodes) {
    sdc->graph->g.getData(n).totalPinC = pinC;
  }
}

static void readSDC(FileReader& fRd, SDC* sdc) {
  for (std::string token = fRd.nextToken(); token != "";
       token             = fRd.nextToken()) {
    if ("create_clock" == token) {
      createClock(fRd, sdc);
    } else if ("set_max_delay" == token) {
      setMaxDelay(fRd, sdc);
    } else if ("set_driving_cell" == token) {
      setDrivingCell(fRd, sdc);
    } else if ("set_load") {
      setLoad(fRd, sdc);
    }
  }
}

static void setDefaultValue(SDC* sdc) {
  sdc->targetDelay = std::numeric_limits<float>::infinity();
}

void SDC::setConstraints(std::string inName) {
  setDefaultValue(this);

  if (!inName.empty()) {
    std::cout << "Use the values in " << inName << ".\n";

    char delimiters[] = {'(', ')', ',', ':', ';', '/',  '#',
                         '[', ']', '{', '}', '*', '\"', '\\'};

    char separators[] = {' ', '\t', '\n', ','};

    FileReader fRd(inName, delimiters, sizeof(delimiters), separators,
                   sizeof(separators));
    readSDC(fRd, this);
  } else {
    std::cout << "Use 0.0, default value.\n";
  }
}

SDC::SDC(CellLib* lib, VerilogModule* m, CircuitGraph* g) {
  cellLib = lib;
  vModule = m;
  graph   = g;
}

SDC::~SDC() {}
