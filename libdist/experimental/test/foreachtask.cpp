#include "galois/ForeachTask.h"
#include "galois/Runtime/DistributedStructs.h"
#include <vector>

using namespace std;
using namespace galois::Runtime;

struct R : public galois::runtime::Lockable {
   int i;

   R() { i = 0; }

   void add(int v) {
      i += v;
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

   galois::setActiveThreads(4);

   // check the task id and decide if the following should be executed
   galois::for_each_begin();

   vector<int> myvec;
   typedef vector<int>::iterator IterTy;
   f1 f;
   for (int i=1; i<=40; i++) myvec.push_back(i);
   galois::for_each_task<IterTy,f1> (myvec.begin(), myvec.end(), f);
   printf ("final output: %d\n", f.r->i);

   master_terminate();

   return 0;
}
