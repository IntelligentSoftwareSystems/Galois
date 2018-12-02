#include "AsyncTimingArcSet.h"

#include <cassert>

void AsyncTimingArcSet::print(std::ostream& os) {
  for (auto& i: requiredArcs) {
    auto ip = i.first.first;
    auto iRise = i.first.second;
    auto op = i.second.first;
    auto oRise = i.second.second;

    auto strRF =
        [&] (bool isRise) -> std::string {
          return (isRise) ? "r" : "f";
        };

    os << "(" << ip->gate->name << "/" << ip->name << ", " << strRF(iRise) << ")";
    os << " -> ";
    os << "(" << op->gate->name << "/" << op->name << ", " << strRF(oRise) << ")";
    if (tickedArcs.count(i)) {
      os << " ticked";
    }
    os << std::endl;
  }
}

void AsyncTimingArcCollection::setup() {
  for (auto& i: v.modules) {
    modules[i.second] = new AsyncTimingArcSet;
  }
}

void AsyncTimingArcCollection::clear() {
  for (auto& i: modules) {
    delete i.second;
  }
}

void AsyncTimingArcCollection::parse(std::string inName) {
  AsyncTimingArcParser parser(*this);
  parser.parse(inName);
}

void AsyncTimingArcCollection::print(std::ostream& os) {
  for (auto& i: modules) {
    i.second->print(os);
  }
}

void AsyncTimingArcParser::tokenizeFile(std::string inName) {
  std::vector<char> delimiters = {};
  std::vector<char> separators = {' ', '\t', '\n', '\r'};
  std::vector<std::string> comments = {};

  Tokenizer tokenizer(delimiters, separators, comments);
  tokens = tokenizer.tokenize(inName);
  curToken = tokens.begin();
}

void AsyncTimingArcParser::parseTimingArc(VerilogModule* m) {
  Token wireName_1 = *curToken;
  bool inRise = (wireName_1[0] != '~');
  if (!inRise) {
    wireName_1 = wireName_1.substr(1);
  }

  Token wireName_2 = *(curToken + 1);
  bool outRise = (wireName_2[0] != '~');
  if (!outRise) {
    wireName_2 = wireName_2.substr(1);
  }

  bool isTicked = ("1" == *(curToken + 2));

  // find the gate output pin from wire_2
  auto wire_2 = m->findWire(wireName_2);
  VerilogPin* op = nullptr;
  for (auto& p: wire_2->pins) {
    if (p->gate) {
      // find the cell pin cp for the gate pin p
      auto cp = c.lib.findCell(p->gate->cellType)->findCellPin(p->name);
      if (OUTPUT == cp->pinDir()) {
        op = p; // the gate output pin is found
        break;
      }
    }
  }
  assert(op);

  // add arcs to op from p in wire_1, p & op in the same gate
  auto arcSet = c.findArcSetForModule(m);
  auto wire_1 = m->findWire(wireName_1);
  for (auto& p: wire_1->pins) {
    if (p->gate == op->gate) {
      arcSet->addRequiredArc(p, inRise, op, outRise);
      if (isTicked) {
        arcSet->addTickedArc(p, inRise, op, outRise);
      }
    }
  }

  // consume tokens for this timing arc
  curToken += 3;
}

// process the only top-level module for now
void AsyncTimingArcParser::parse(std::string inName) {
  tokenizeFile(inName);
  while (!isEndOfTokenStream()) {
    parseTimingArc(*(c.v.roots.begin()));
  }
}
