#include "Galois/ForeachTask.h"
#include "Galois/Runtime/DistributedStructs.h"
#include <vector>

using namespace std;
using namespace Galois::Runtime;

struct R : public Galois::Runtime::Lockable {
   int i;

   R() { i = 0; }

   void add(int v) {
      i += v;
      return;
   }
};


struct f1 {
   dptr<R> r;

   void operator()(int& data, Galois::UserContext<int>& lwl) {
      r->add(data);
 //printf ("data: %d\t sum so far: %d\n", data, r->i);
      return;
   }

   void marshall (char *data) {
      memcpy (data, this, sizeof(f1));
      return;
   }

   static void rstart (char *f, int count, char *data) {
      using namespace Galois::Runtime::WorkList;
      typedef LIFO<int,true> chunk;
  //  typedef ChunkedFIFO<16> chunk;
      int *beg = (int*)data;
      int *end = (beg + count);
      f1 *fptr = (f1*)f;
      Galois::for_each<chunk>(beg,end,*fptr);
      return;
   }
};

int main(int argc, char *argv[])
{
 /*
   int  rc;

   rc = MPI_Init(&argc,&argv);
   if (rc != MPI_SUCCESS) {
     printf ("Error starting MPI program. Terminating.\n");
     MPI_Abort(MPI_COMM_WORLD, rc);
   }
  */

   Galois::setActiveThreads(4);

   // check the task id and decide if the following should be executed
   Galois::for_each_begin();

   vector<int> myvec;
   typedef vector<int>::iterator IterTy;
   f1 f;
   for (int i=1; i<=40; i++) myvec.push_back(i);
   Galois::for_each_task<IterTy,f1> (myvec.begin(), myvec.end(), f);
   printf ("final output: %d\n", f.r->i);

   master_terminate();

   return 0;
}
