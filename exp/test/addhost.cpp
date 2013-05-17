#include <vector>
#include "Galois/Galois.h"
#include "Lonestar/BoilerPlate.h"
#include "Galois/Runtime/DistSupport.h"

using namespace std;
using namespace Galois::Runtime;
using namespace Galois::Runtime::Distributed;

typedef vector<int>::iterator IterTy;

struct R : public Galois::Runtime::Lockable {
   int i;

  R() :i(0) {}

  void add(int v) {
    std::cerr << "In Host " << networkHostID << " and thread " << LL::getTID() << " processing number " << v << " old value " << i << "\n";
    i += v;
    return;
  }

  typedef int tt_has_serialize;
  void serialize(Galois::Runtime::SerializeBuffer& s) const {
    gSerialize(s,i);
  }
  void deserialize(Galois::Runtime::DeSerializeBuffer& s) {
    gDeserialize(s,i);
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
  void serialize(Galois::Runtime::SerializeBuffer& s) const {
    gSerialize(s,r);
  }
  void deserialize(Galois::Runtime::DeSerializeBuffer& s) {
    gDeserialize(s,r);
  }
};

static const char *name = "addhost distributed testcase";
static const char *desc = "sum of 40 numbers using distributed host";
static const char *url  = "addhost";

int main(int argc, char *argv[])
{
   LonestarStart(argc, argv, name, desc, url);

   // check the host id and initialise the network
   Galois::Runtime::networkStart();

   vector<int> myvec;
   typedef Galois::WorkList::LIFO<int,true> chunk;
   R r;
   f1 f(&r);
   for (int i=1; i<=40; i++) myvec.push_back(i);

   std::cerr << "stating\n";

   Galois::for_each<chunk,IterTy,f1> (myvec.begin(), myvec.end(), f);
   std::cerr << "sum is " << f.r->i << "\n";
   std::cerr << "sum should be " << std::accumulate(myvec.begin(), myvec.end(), 0) << "\n";
   // master_terminate();
   Galois::Runtime::networkTerminate();

   return 0;
}
