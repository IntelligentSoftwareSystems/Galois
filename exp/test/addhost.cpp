#include <vector>
#include "Galois/Galois.h"
#include "Lonestar/BoilerPlate.h"
#include "Galois/Runtime/DistSupport.h"

using namespace std;
using namespace Galois::Runtime;

typedef vector<int>::iterator IterTy;

struct R : public Galois::Runtime::Lockable {
   int i;

  R() :i(0) {}

  void add(int v) {
    std::cerr << "In Host " << NetworkInterface::ID << " and thread " << LL::getTID() << " processing number " << v << " old value " << i << "\n";
    i += v;
    return;
  }

  typedef int tt_has_serialize;
  void deserialize(RecvBuffer& buf) {
    gDeserialize(buf, i);
  }
  void serialize(SendBuffer& buf) const {
    gSerialize(buf, i);
  }
};

struct f1 {
  gptr<R> r;
  
  f1(R* _r = nullptr) :r(_r) {}
  
  void operator()(int& data, Galois::UserContext<int>& lwl) {
    //    r.dump(std::cerr);
    acquire(r, Galois::ALL);
    r->add(data);
    return;
  }
};

static const char *name = "addhost distributed testcase";
static const char *desc = "sum of 40 numbers using distributed host";
static const char *url  = "addhost";

int main(int argc, char *argv[])
{
  Galois::StatManager M;
  LonestarStart(argc, argv, name, desc, url);
  
  vector<int> myvec;
  R r;
  f1 f(&r);
  for (int i=1; i<=40; i++) myvec.push_back(i);
  
  static_assert(Galois::Runtime::is_serializable<R>::value, "R not serializable");

  int i1{42}, i2;
  SendBuffer sbuf;
  Galois::Runtime::gSerialize(sbuf, f.r, i1);
  gptr<R> e;
  RecvBuffer rbuf(std::move(sbuf));
  Galois::Runtime::gDeserialize(rbuf, e, i2);
  std::cerr << f.r << " " << e << " " << i1 << " " << i2 << "\n";

  std::cerr << "stating\n";
  
  Galois::for_each(myvec.begin(), myvec.end(), f, Galois::wl<Galois::WorkList::LIFO<>>());
  acquire(f.r, Galois::ALL);
  auto val_is = f.r->i, val_sb = std::accumulate(myvec.begin(), myvec.end(), 0);
  std::cerr << "sum is " << val_is << "\n";
  std::cerr << "sum should be " << val_sb << "\n";
  // master_terminate();
  getSystemNetworkInterface().terminate();
  
  return val_is != val_sb; // false is success
}
