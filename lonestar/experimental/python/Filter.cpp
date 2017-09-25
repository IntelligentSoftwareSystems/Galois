#include "Filter.h"
#include "galois/Timer.h"
#include "galois/Bag.h"

#include <regex>
#include <string>
#include <iostream>

template<bool isFullMatch>
struct RegExMatch {
  Graph& g;
  galois::InsertBag<GNode>& bag;
  KeyAltTy key;
  std::regex regex;

  RegExMatch(Graph& g, galois::InsertBag<GNode>& bag, KeyAltTy key, std::regex regex)
    : g(g), bag(bag), key(key), regex(regex) 
  {}

  void operator()(GNode n) {
    auto& data = g.getData(n);
    auto it = data.attr.find(key);
    // no such a key
    if (it == data.attr.end()) {
      return;
    }
    // find a full match
    if (isFullMatch && std::regex_match(it->second, regex)) {
//      std::cout << "full match: " << it->second << " in " << n << std::endl;
      bag.push_back(n);
    }
    // find a subsequence match
    if (!isFullMatch && std::regex_search(it->second, regex)) {
//      std::cout << "subsequence match: " << it->second << " in " << n << std::endl;
      bag.push_back(n);
    }
  }
};

NodeList filterNode(Graph *g, const KeyAltTy key, const ValAltTy value, bool isFullMatch) {
//  galois::StatManager statManager;
  galois::InsertBag<GNode> bag;

//  galois::StatTimer T;
//  T.start();

//  std::cout << "regex = " << value << std::endl;
  std::regex regex(value);

  if (isFullMatch) {
    galois::do_all_local(*g, RegExMatch<true>{*g, bag, key, regex}, galois::steal<true>());
  } else {
    galois::do_all_local(*g, RegExMatch<false>{*g, bag, key, regex}, galois::steal<true>());
  }

//  T.stop();

  int num = std::distance(bag.begin(), bag.end());
  NodeList l = createNodeList(num);
  int i = 0;
  for (auto n: bag) {
    l.nodes[i++] = n;
  }
  return l;
}

