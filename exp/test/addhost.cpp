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
  
  // check the host id and initialise the network
  getSystemNetworkInterface().start();
  
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
  f.r.dump(std::cerr);
  e.dump(std::cerr);
  std::cerr << " " << i1 << " " << i2 << "\n";

  std::cerr << "stating\n";
  
  Galois::for_each(myvec.begin(), myvec.end(), f, Galois::wl<Galois::WorkList::LIFO<>>());
  std::cerr << "sum is " << f.r->i << "\n";
  std::cerr << "sum should be " << std::accumulate(myvec.begin(), myvec.end(), 0) << "\n";
  // master_terminate();
  getSystemNetworkInterface().terminate();
  
  return 0;
}
