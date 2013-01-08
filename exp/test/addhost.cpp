#include <vector>
#include "Galois/Galois.h"
#include "Galois/Runtime/gptr.h"

using namespace std;
using namespace Galois::Runtime;

typedef vector<int>::iterator IterTy;

struct R {
   int i;

   R() { i = 0; }

   void add(int v) {
      i += v;
      return;
   }
};

struct f1 {
   gptr<R> r;

   void operator()(int& data, Galois::UserContext<int>& lwl) {
      r->add(data);
      return;
   }
};

int main(int argc, char *argv[])
{
   // Galois::setActiveThreads(4);

   // check the host id and initialise the network
   Galois::Runtime::Distributed::networkStart();

   vector<int> myvec;
   f1 f;
   for (int i=1; i<=40; i++) myvec.push_back(i);

   Galois::for_each<IterTy,f1> (myvec.begin(), myvec.end(), f);
   printf ("sum is %d\n", f.r->i);

   // master_terminate();
   Galois::Runtime::Distributed::networkTerminate();

   return 0;
}
