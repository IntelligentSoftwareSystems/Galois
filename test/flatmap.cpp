#include "Galois/FlatMap.h"

#include <iostream>

int main() {

  Galois::flat_map<int, double> m;
  Galois::flat_map<int, double> m2(m);
  Galois::flat_map<int, double> m3 {{10,0},{20,0}};
  Galois::flat_map<int, double> m4(m3.begin(), m3.end());
  m2 = m3;
  m3 = std::move(m2);

  m[0] = 0.1;
  m[1] = 1.2;
  m[4] = 2.3;
  m[3] = 3.4;

  m[3] += 10.0;

  m.insert(std::make_pair(5, 4.5));
  m.insert(m4.begin(), m4.end());

  std::cout << "10 == " << m.find(10)->first << "\n";

  m.erase(m.find(10));
  m.erase(1);

  std::cout << m.size() << " " << m.empty() << " " << m.max_size() << "\n";
  m.swap(m3);
  std::cout << m.size() << " " << m.empty() << " " << m.max_size() << "\n";
  m.clear();
  std::cout << m.size() << " " << m.empty() << " " << m.max_size() << "\n";
  m.swap(m3);
  std::cout << m.size() << " " << m.empty() << " " << m.max_size() << "\n";

  std::cout << m.at(0) << " " << m.count(0) << " " << (m == m) << "\n";

  for (auto ii = m.begin(), ee = m.end(); ii != ee; ++ii)
    std::cout << ii->first << " " << ii->second << " ";
  std::cout << "\n";

  for (auto ii = m.cbegin(), ee = m.cend(); ii != ee; ++ii)
    std::cout << ii->first << " " << ii->second << " ";
  std::cout << "\n";

  for (auto ii = m.rbegin(), ee = m.rend(); ii != ee; ++ii)
    std::cout << ii->first << " " << ii->second << " ";
  std::cout << "\n";

  for (auto ii = m.crbegin(), ee = m.crend(); ii != ee; ++ii)
    std::cout << ii->first << " " << ii->second << " ";
  std::cout << "\n";

  return 0;
}
