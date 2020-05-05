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

#include "Filter.h"
#include "galois/Timer.h"
#include "galois/Bag.h"

#include <regex>
#include <string>
#include <iostream>

template <bool isFullMatch>
struct RegExMatch {
  Graph& g;
  galois::InsertBag<GNode>& bag;
  KeyAltTy key;
  std::regex regex;

  RegExMatch(Graph& g, galois::InsertBag<GNode>& bag, KeyAltTy key,
             std::regex regex)
      : g(g), bag(bag), key(key), regex(regex) {}

  void operator()(GNode n) {
    auto& data = g.getData(n);
    auto it    = data.attr.find(key);
    // no such a key
    if (it == data.attr.end()) {
      return;
    }
    // find a full match
    if (isFullMatch && std::regex_match(it->second, regex)) {
      //      std::cout << "full match: " << it->second << " in " << n <<
      //      std::endl;
      bag.push_back(n);
    }
    // find a subsequence match
    if (!isFullMatch && std::regex_search(it->second, regex)) {
      //      std::cout << "subsequence match: " << it->second << " in " << n <<
      //      std::endl;
      bag.push_back(n);
    }
  }
};

NodeList filterNode(Graph* g, const KeyAltTy key, const ValAltTy value,
                    bool isFullMatch) {
  //  galois::StatManager statManager;
  galois::InsertBag<GNode> bag;

  //  galois::StatTimer T;
  //  T.start();

  //  std::cout << "regex = " << value << std::endl;
  std::regex regex(value);

  if (isFullMatch) {
    galois::do_all(*g, RegExMatch<true>{*g, bag, key, regex}, galois::steal());
  } else {
    galois::do_all(*g, RegExMatch<false>{*g, bag, key, regex}, galois::steal());
  }

  //  T.stop();

  int num    = std::distance(bag.begin(), bag.end());
  NodeList l = createNodeList(num);
  int i      = 0;
  for (auto n : bag) {
    l.nodes[i++] = n;
  }
  return l;
}
