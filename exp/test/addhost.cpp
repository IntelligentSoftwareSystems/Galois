#include <vector>
#include "Galois/Galois.h"
#include "Galois/Runtime/DistSupport.h"

using namespace std;
using namespace Galois::Runtime;
using namespace Galois::Runtime::Distributed;

typedef vector<int>::iterator IterTy;

struct R : public Galois::Runtime::Lockable {
   int i;

   R() { i = 0; }

   void add(int v) {
     //printf ("In Host %u: processing number %d\n", networkHostID, v);
      i += v;
      return;
   }

  // serialization functions
  typedef int tt_has_serialize;
  void serialize(Galois::Runtime::Distributed::SerializeBuffer& s) const {
    s.serialize(i);
  }
  void deserialize(Galois::Runtime::Distributed::DeSerializeBuffer& s) {
    s.deserialize(i);
  }
};

struct f1 {
  gptr<R> r;

  f1(R* _r = nullptr) :r(_r) {}

   void operator()(int& data, Galois::UserContext<int>& lwl) {
      r->add(data);
      return;
   }

	// serialization functions
	typedef int tt_has_serialize;
  void serialize(Galois::Runtime::Distributed::SerializeBuffer& s) const {
    s.serialize(r);
  }
  void deserialize(Galois::Runtime::Distributed::DeSerializeBuffer& s) {
    s.deserialize(r);
  }
};

int main(int argc, char *argv[])
{
  //Galois::setActiveThreads(4);

   // check the host id and initialise the network
   Galois::Runtime::Distributed::networkStart();

   vector<int> myvec;
   typedef WorkList::LIFO<int,true> chunk;
   R r;
   f1 f(&r);
   for (int i=1; i<=40; i++) myvec.push_back(i);

   Galois::for_each<chunk,IterTy,f1> (myvec.begin(), myvec.end(), f);
   printf ("sum is %d\n", f.r->i);

   // master_terminate();
   Galois::Runtime::Distributed::networkTerminate();

   return 0;
}
