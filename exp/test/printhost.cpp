#include "Galois/Galois.h"
#include <vector>

using namespace std;
using namespace Galois::Runtime;

typedef vector<int>::iterator IterTy;

struct f1 {
  void operator()(int& data, Galois::UserContext<int>& lwl) {
    printf ("%d,%d ", Galois::Runtime::networkHostID, data);
    return;
  }
};

int main(int argc, char *argv[])
{
   // Galois::setActiveThreads(4);

   // check the host id and initialise the network
   Galois::Runtime::networkStart();

   vector<int> myvec;
   f1 f;
   for (int i=1; i<=40; i++) myvec.push_back(i);

   printf ("Data:");
   Galois::for_each<IterTy,f1> (myvec.begin(), myvec.end(), f);
   printf ("\n");

   // master_terminate();
   Galois::Runtime::networkTerminate();

   return 0;
}
