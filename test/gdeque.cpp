#include <iostream>

#include "Galois/gdeque.h"

using namespace std;
using namespace Galois;

int main() {
  gdeque<int> mydeque;
  for (int i = 0; i < 10; ++i)
    mydeque.push_back (i);

  cout << "Start size of mydeque is " << int(mydeque.size());
  cout << "\nBy Iter:";
  for (gdeque<int>::iterator it=mydeque.begin(); it!=mydeque.end(); ++it)
    cout << " " << *it;

  cout << "\nSize of mydeque is " << int(mydeque.size());
  cout << "\nPopping front out the elements in mydeque:";
  while (!mydeque.empty())
  {
    cout << " " << mydeque.front();
    mydeque.pop_front();
  }
  cout << "\nSize of mydeque is " << int(mydeque.size());

  for (int i = 0; i < 10; ++i)
    mydeque.push_back(i);
  cout << "\nSize of mydeque is " << int(mydeque.size());
  cout << "\nPopping back out the elements in mydeque:";
  while (!mydeque.empty())
  {
    cout << " " << mydeque.back();
    mydeque.pop_back();
  }
  cout << "\nFinal size of mydeque is " << int(mydeque.size()) << "\n";

  return 0;
}
