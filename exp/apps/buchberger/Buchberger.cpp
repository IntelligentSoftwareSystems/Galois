/** Buchberger Algorithm -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2012, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 *
 * @section Description
 *
 * Buchberger algorithm for computing Groebner basis. Based off of sage
 * implementation in sage.rings.polynomial.toy_buchberger by Martin Albrecht
 * and Marshall Hampton.
 *
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 */

#include "Galois/Galois.h"
#include "Galois/Bag.h"
#include "Galois/Statistic.h"
#include "Galois/ParallelSTL.h"
#include "llvm/Support/CommandLine.h"
#include "Lonestar/BoilerPlate.h"

#include <gmpxx.h>
//#include <givaro/givgfq.h>
//#include <givaro/StaticElement.h>

#include <boost/spirit/include/phoenix_core.hpp>
#include <boost/spirit/include/phoenix_operator.hpp>
#include <boost/spirit/include/phoenix_fusion.hpp>
#include <boost/spirit/include/phoenix_stl.hpp>
#include <boost/spirit/include/phoenix_object.hpp>
#include <boost/spirit/include/support_istream_iterator.hpp>
#include <boost/spirit/include/qi.hpp>
#include <boost/fusion/include/adapt_struct.hpp>
#include <boost/bind.hpp>
#include <boost/utility.hpp>

#include <fstream>
#include <map>
#include <numeric>
#include <algorithm>

namespace cll = llvm::cl;

static const char* name = "Buchberger Algorithm";
static const char* desc = "Generates Groebner basis for polynomial ideal using Buchberger Algorithm";
static const char* url = 0;

enum MonomialOrder {
  lex,
  grevlex
};

static cll::opt<std::string> filename(cll::Positional, cll::desc("<input file>"), cll::Required);
static cll::opt<MonomialOrder> monomialOrder(cll::desc("Monomial order:"),
    cll::values(
      clEnumVal(lex, "lexicographic"),
      clEnumVal(grevlex, "graded reverse lexicographic"),
      clEnumValEnd), cll::init(grevlex));

#if 0
//! Unfortunate pun
struct GaloisField {
  typedef Givaro::GFqDom<long> Field;
  typedef Field::Element element_type;
  Field F;

  GaloisField(): F(2, 1) { }

  void init() { }

  void inverse(element_type& a) const {
    F.invin(a);
  }
  
  void add(element_type& a, const element_type& b) const {
    F.addin(a, b);
  }

  void subtract(element_type& a, const element_type& b) const {
    F.subin(a, b);
  }

  void negate(element_type& a) const {
    F.negin(a);
  }

  void divide(element_type& a, const element_type& b) const {
    F.divin(a, b);
  }

  void multiply(element_type& a, const element_type& b) const {
    F.mulin(a, b);
  }

  void assign(element_type& a, const element_type& b) const {
    F.assign(a, b);
  }

  void assignOne(element_type& a) const {
    F.assign(a, F.one);
  }

  int sign(const element_type& a) const {
    return neq(a, 0) ? 1 : 0;
  }

  bool neq(const element_type& a, const element_type& b) const {
    return a != b;
  }

  void write(std::ostream& out, const element_type& a) const {
    out << a;
  }
};
#endif

struct RationalField {
  typedef mpq_class element_type;

  void init() { }

  void inverse(element_type& a) const {
    a = 1 / a;
  }
  
  void add(element_type& a, const element_type& b) const {
    a += b;
  }

  void subtract(element_type& a, const element_type& b) const {
    a -= b;
  }

  void negate(element_type& a) const {
    a = -a;
  }

  void divide(element_type& a, const element_type& b) const {
    a /= b;
  }

  void multiply(element_type& a, const element_type& b) const {
    a *= b;
  }

  void assign(element_type& a, const element_type& b) const {
    a = b;
  }

  void assignOne(element_type& a) const {
    a = 1;
  }

  int sign(const element_type& a) const {
    return sgn(a);
  }

  bool neq(const element_type& a, const element_type& b) const {
    return a != b;
  }

  void write(std::ostream& out, const element_type& a) const {
    out << a.get_str();
  }
};

typedef RationalField Field;
//typedef GaloisField Field;
typedef Field::element_type number;

Field TheField;

template<class Order,class Alloc> class Ring;

//! Weights to ease vectorization of lexicographic sorting
static int powersOfTwo[16] __attribute__((__aligned__(16)));

struct Term {
  typedef char exp_type; 
  typedef char __attribute__((__aligned__(16))) * __restrict__ exp_ptr_type;

private:
  template<class Order,class Alloc> friend class Ring;

  Field::element_type m_coef;
  exp_ptr_type m_exps;
  int m_totalDegree;

public:
  Term(): m_coef(1), m_exps(NULL), m_totalDegree(0) { }

  Field::element_type& coef() { return m_coef; }
  const Field::element_type& coef() const { return m_coef; }

  exp_type& exp(int i) { return m_exps[i]; }
  const exp_type& exp(int i) const { return m_exps[i]; }
  
  int totalDegree() const { return m_totalDegree; }
  
  exp_ptr_type const & exps() const { return m_exps; }
  exp_ptr_type& exps() { return m_exps; }

  //! Compute LCM
  template<class R>
  void makeLcm(const Term& a, const Term& b, R& ring) {
    // Vectorized.
    // GCC 4.7 vectorizer prefers (1) over (2)
    int N = ring.numVars();
    for (int i = 0; i < N; ++i) { // (1)
    //for (int i = 0; i < ring.numVars(); ++i) { // (2)
      m_exps[i] = std::max(a.m_exps[i], b.m_exps[i]);
    }
  }

  template<class R>
  bool equals(const Term& b, R& ring) {
    // TODO vectorize
    int N = ring.numVars();
    for (int i = 0; i < N; ++i) {
      if (b.exp(i) != exp(i))
        return false;
    }
    return true;
  }

  //! a | b ?
  template<class R>
  bool divides(const Term& b, const R& ring) const {
    // TODO vectorize
    for (int i = 0; i < ring.numVars(); ++i) {
      if (b.exp(i) < exp(i))
        return false;
    }
    return true;
  }

  //! Relatively prime?
  template<class R>
  bool relPrime(const Term& b, const R& ring) const {
    // TODO vectorize
    for (int i = 0; i < ring.numVars(); ++i) {
      if (exp(i) && b.exp(i))
        return false;
    }
    return true;
  }
};

//! Sequence of terms
class Poly {
  // TODO: more bucketed representations!!!
  typedef std::vector<Term*> Terms;
  Terms terms;
  int m_totalDegree;

public:
  typedef Terms::iterator iterator;
  typedef Terms::const_iterator const_iterator;

  Poly(): m_totalDegree(0) { }

  int totalDegree() const { return m_totalDegree; }

  const Term* head() const {
    return terms.front();
  }
  
  Term* head() {
    return terms.front();
  }
  
  const_iterator begin() const {
    return terms.begin();
  }
  
  const_iterator end() const {
    return terms.end();
  }
  
  iterator begin() {
    return terms.begin();
  }
  
  iterator end() {
    return terms.end();
  }

  void push(Term* t) {
    terms.push_back(t);
    m_totalDegree += t->totalDegree();
  }

  bool empty() const { return terms.empty(); }

  //! f = f * s
  template<class R>
  void scaleBy(const Field::element_type& s, R& ring) {
    for (iterator ii = begin(), ei = end(); ii != ei; ++ii) {
      Term* t = *ii;
      TheField.multiply(t->coef(),  s);
    }
  }

  //! f = f * a/b
  template<class R>
  void scaleBy(const Term& a, const Term& b, R& ring) {
    Field::element_type s;
    TheField.assign(s, a.coef());
    TheField.divide(s, b.coef());
   
    m_totalDegree = 0;

    for (iterator ii = begin(), ei = end(); ii != ei; ++ii) {
      Term* t = *ii;
      // Vectorized.
      for (int i = 0; i < ring.numVars(); ++i) {
        assert(a.exp(i) >= b.exp(i));
        t->exp(i) += a.exp(i) - b.exp(i);
      }
      TheField.multiply(t->coef(),  s);
      ring.generateTotalDegree(*t);
      m_totalDegree += t->totalDegree();
    }
  }
};

class PolySet {
  // TODO: better allocator
  typedef std::vector<Poly*> Polys;
  
  Polys polys;
public:
  typedef Polys::iterator iterator;
  typedef Polys::const_iterator const_iterator;

  void push(Poly* p) {
    polys.push_back(p);
  }

  iterator begin() {
    return polys.begin();
  }

  iterator end() {
    return polys.end();
  }

  const_iterator begin() const {
    return polys.begin();
  }

  const_iterator end() const {
    return polys.end();
  }
};

template<class Order, class Alloc=std::allocator<char> >
class Ring: private boost::noncopyable {
private:
  Order order;
  Alloc m_alloc;
  int m_numVars;
  std::list<Poly*> polys;
  std::list<Term*> terms;

  //! Sizeof Term plus padding and space for exps
  size_t sizeofTerm() const {
    return sizeof(Term) + 15 + m_numVars * sizeof(Term::exp_type);
  }

  typedef Term::exp_type* exps_ptr_t;

  //! Find 16-byte aligned address after ptr
  exps_ptr_t expsPtr(const Term* ptr) const {
    return reinterpret_cast<exps_ptr_t>(((uintptr_t)(ptr + 1) + 15) & ~0x0F);
  }

public:
  const static int kBlockSize = 16 / sizeof(Term::exp_type);

  template<class AllocNew>
  struct realloc { typedef Ring<Order,AllocNew> other; };

  Ring(int numVars, const Alloc& alloc=Alloc()): m_alloc(alloc) { 
    m_numVars = ((numVars + kBlockSize - 1) / kBlockSize) * kBlockSize;
  }

  ~Ring() {
    for (std::list<Poly*>::iterator ii = polys.begin(), ei = polys.end(); ii != ei; ++ii) {
      (*ii)->~Poly();
      m_alloc.deallocate(reinterpret_cast<char*>(*ii), sizeof(Poly));
    }
    for (std::list<Term*>::iterator ii = terms.begin(), ei = terms.end(); ii != ei; ++ii) {
      (*ii)->~Term();
      m_alloc.deallocate(reinterpret_cast<char*>(*ii), sizeofTerm());
    }
  }

  // TODO
  Ring tempRing() const {
    return *this;
  }

  //! For ease of vectorization, this will always be a multiple of blockSize
  int numVars() const { return m_numVars; }

  bool gt(const Term& a, const Term& b) const {
    return order.gt(a, b, *this);
  }

  Poly& makePoly(const Poly& p = Poly()) {
    Poly* ptr = reinterpret_cast<Poly*>(m_alloc.allocate(sizeof(Poly)));
    new (ptr) Poly();
    for (Poly::const_iterator ii = p.begin(), ei = p.end(); ii != ei; ++ii) {
      const Term* t = *ii;
      ptr->push(&makeTerm(*t));
    }
    polys.push_back(ptr);
    return *ptr;
  }

  Term& makeTerm(const Term& t = Term()) {
    Term* ptr = reinterpret_cast<Term*>(m_alloc.allocate(sizeofTerm()));
    new (ptr) Term(t);
    ptr->m_exps = expsPtr(ptr);

    if (t.exps() != NULL) {
      memcpy(ptr->m_exps, t.exps(), m_numVars * sizeof(Term::exp_type));
    } else {
      memset(ptr->m_exps, 0, m_numVars * sizeof(Term::exp_type));
    }
    terms.push_back(ptr);
    return *ptr;
  }

  void generateTotalDegree(Term& t) {
    // Vectorized. 
    // GCC 4.7 vectorizer prefers (1) over (2)
    int sum = 0;
    Term::exp_ptr_type ii = t.exps();
    for (int i = 0; i < numVars(); ++i) // (1)
      sum += ii[i];
    t.m_totalDegree = sum;
    //std::accumulate(t.exps(), t.exps() + numVars(), 0, std::plus<int>()); // (2)
  }

  void write(std::ostream& out, const Term& t, const std::vector<std::string>* idMap = NULL) const {
    if (TheField.sign(t.coef()) >= 0)
      out << "+";
    TheField.write(out, t.coef());

    for (int i = 0; i < numVars(); ++i) {
      int exp = t.exp(i);
      if (exp == 0)
        continue;
      out << " ";
      if (idMap)
        out << (*idMap)[i];
      else
        out << "x" << i;
      if (exp > 1)
        out << "^" << exp;
    }
  }

  void write(std::ostream& out, const Term* t, const std::vector<std::string>* idMap = NULL) const {
    write(out, *t, idMap);
  }

  void write(std::ostream& out, const Poly& p, const std::vector<std::string>* idMap = NULL) const {
    for (Poly::const_iterator ii = p.begin(), ei = p.end(); ii != ei; ++ii) {
      write(out, *ii, idMap);
    }
  }

  void write(std::ostream& out, const PolySet& polys, const std::vector<std::string>* idMap = NULL) const {
    for (PolySet::const_iterator ii = polys.begin(), ei = polys.end(); ii != ei; ++ii) {
      write(out, **ii, idMap);
      if (ii + 1 != ei)
        out << ", ";
    }
  }

  //! Rank under Buchberger's normal selection strategy
  int normalRank(const Poly& f, const Poly& g) const {
    const Term* fh = f.head();
    const Term* gh = g.head();
    int retval = 0;

    // Vectorized.
    for (int i = 0; i < numVars(); ++i) {
      retval += std::max(fh->exp(i), gh->exp(i));
    }

    return retval;
  }

  //! Rank pairs using sugar strategy
  int rankPair(const Poly& f, const Poly& g) const {
    const Term* fh = f.head();
    const Term* gh = g.head();
    int retval = std::max(f.totalDegree() - fh->totalDegree(), g.totalDegree() - gh->totalDegree());
    int n = normalRank(f, g);
    // prefer pairs with least sugar component, breaking ties with normalRank
    int sugar = retval + n;
    return (sugar << 8) | (n & 0xFF);
  }
};

class LexOrder {
  template<class R>
  bool gtSimple(const Term& a, const Term& b, const R& ring) const {
    for (int i = 0; i < ring.numVars(); ++i) {
      if (a.exp(i) < b.exp(i))
        return false;
      else if (a.exp(i) > b.exp(i))
        return true;
    }
    return false;
  }

  template<class R>
  bool gtVectorized(const Term& a, const Term& b, const R& ring) const {
    Term::exp_ptr_type aa = a.exps();
    Term::exp_ptr_type bb = b.exps();
    int* pp = powersOfTwo;

    for (int block = 0; block < ring.numVars(); block += R::kBlockSize) { 
      // Sigh. In GCC 4.7 this inner loop (1) is not vectorized because of
      // the condition (2)
      int smaller = 0;
      int larger = 0;
      for (int i = 0; i < R::kBlockSize; ++i) { // (1)
        int idx = block + i;
        smaller += aa[idx] < bb[idx] ? pp[R::kBlockSize - i] : 0;
        larger += aa[idx] > bb[idx] ? pp[R::kBlockSize - i] : 0;
      }
      if (smaller == larger) // (2)
        continue;
      return larger > smaller;
    }

    return false;
  }

public:
  template<class R>
  bool gt(const Term& a, const Term& b, const R& ring) const {
    return gtVectorized(a, b, ring);
  }
};

class GrevlexOrder {
  template<class R>
  bool gtSimple(const Term& a, const Term& b, const R& ring) const {
    if (a.totalDegree() != b.totalDegree())
      return a.totalDegree() > b.totalDegree();

    for (int i = ring.numVars() - 1; i >= 0; --i) {
      if (a.exp(i) < b.exp(i))
        return true;
      else if (a.exp(i) > b.exp(i))
        return false;
    }
    return false;
  }

  template<class R>
  bool gtVectorized(const Term& a, const Term& b, const R& ring) const {
    if (a.totalDegree() != b.totalDegree())
      return a.totalDegree() > b.totalDegree();

    Term::exp_ptr_type aa = a.exps();
    Term::exp_ptr_type bb = b.exps();
    int* pp = powersOfTwo;

    for (int block = ring.numVars() - 1; block >= 0; block -= R::kBlockSize) { 
      // Sigh. In GCC 4.7 this inner loop (1) is not vectorized because of
      // the condition (2)
      int smaller = 0;
      int larger = 0;
      for (int i = 0; i < R::kBlockSize; ++i) { // (1)
        int idx = block - i;
        smaller += aa[idx] < bb[idx] ? pp[R::kBlockSize - i] : 0;
        larger += aa[idx] > bb[idx] ? pp[R::kBlockSize - i] : 0;
      }
      if (smaller == larger) // (2)
        continue;
      return larger > smaller;
    }

    return false;
  }

public:
  template<class R>
  bool gt(const Term& a, const Term& b, const R& ring) const {
    return gtVectorized(a, b, ring);
  }
};

struct PolyPair {
  Poly* a;
  Poly* b;
  Term* lcm;
  int index;
  bool m_useless;
  PolyPair(Poly* _a, Poly* _b, Term* _lcm, int _index = 0): a(_a), b(_b), lcm(_lcm), index(_index), m_useless(false) { }
  bool useless() const { return m_useless; }
  void makeUseless() { m_useless = true; }
};

//! r = a - b
template<class R>
Poly* subtract(const Poly& a, const Poly& b, R& ring) {
  Poly* result = &ring.makePoly();
  Poly::const_iterator aa = a.begin(), ea = a.end();
  Poly::const_iterator bb = b.begin(), eb = b.end();

  while (aa != ea && bb != eb) {
    Term* a = *aa;
    Term* b = *bb;
    Term t;
    if (ring.gt(*a, *b)) {
      t = Term(*a);
      ++aa;
    } else if (ring.gt(*b, *a)) { 
      t = Term(*b);
      TheField.negate(t.coef());
      ++bb;
    } else {
      t = Term(*a);
      TheField.subtract(t.coef(), b->coef());
      ++aa;
      ++bb;
    }
    if (TheField.neq(t.coef(), 0)) {
      result->push(&ring.makeTerm(t));
    }
  }
  for (; aa != ea; ++aa) {
    result->push(*aa);
  }
  for (; bb != eb; ++bb) {
    Term* b = *bb;
    Term t(*b);
    TheField.negate(t.coef());
    assert(TheField.neq(t.coef(), 0));
    result->push(&ring.makeTerm(t));
  }
  return result;
}

//! Compute s-polynomial.
//! spoly(f,g) = f * lcm/LT(f) - g * lcm/LT(g) where lcm = LCM(LM(f), LM(g))
template<class R>
Poly* spoly(const Term& lcm, const Poly& f, const Poly& g, R& ring) {
  // ff = f * lcm/LT(f)
  Poly& ff = ring.makePoly(f);
  ff.scaleBy(lcm, *f.head(), ring);
  // gg = g * lcm/LT(g)
  Poly& gg = ring.makePoly(g);
  gg.scaleBy(lcm, *g.head(), ring);

  return subtract(ff, gg, ring);
};

//! Reduce f with respect to polys.
//! Not canonical unless polys is groebner basis (obvs)
template<class R>
Poly* reduce(const Poly& f, const PolySet& polys, R& ring) {
//  std::cerr << "    "; ring.write(std::cerr, f);

  const Poly* cur = &f;
  Poly::const_iterator ff = cur->begin(), ef = cur->end();

  while (ff != ef) {
    bool reduced = false;

    for (PolySet::const_iterator pp = polys.begin(), ep = polys.end(); pp != ep; ++pp) {
      const Poly* p = *pp;

      // when we are doing inter reduction, i.e., f reduce G when f \in G, skip ourselves
      if (&f == p)
        continue;

      // Right now we never delete terms reduced to zero by interReduce() so ignore them here
      if (p->empty())
        continue;

      const Term* ph = p->head();

      if (!ph->divides(**ff, ring))
        continue;

//      std::cerr << " ("; ring.write(std::cerr, *ph); std::cerr << "|"; ring.write(std::cerr, **ff); std::cerr << ")";
      // cur = cur - p * ff/ph
      Poly& g = ring.makePoly(*p);
      g.scaleBy(**ff, *ph, ring);
      cur = subtract(*cur, g, ring);

      ff = cur->begin();
      ef = cur->end();
//      std::cerr << " => "; ring.write(std::cerr, *cur);
      reduced = true;
      break;
    }
    if (!reduced)
      ++ff;
  }

//  std::cerr << "\n";

  return const_cast<Poly*>(cur);
}

Galois::Statistic bkUpdate("BKUpdate");
Galois::Statistic mfUpdate("MFUpdate");
Galois::Statistic bpUpdate("BPUpdate");

//! Updates basis with h, adds new pairs and marks some previous pairs as useless.
template<class R1,class R2,class Pushable>
void update(Poly* h, PolySet& basis, Galois::InsertBag<PolyPair>& pairs, R1& localRing, R2& ring, Pushable& out) {
  // Gebauer-Moeller criterion B_k
  Term& lcm_t = localRing.makeTerm();
  for (Galois::InsertBag<PolyPair>::iterator ii = pairs.begin(), ei = pairs.end(); ii != ei; ++ii) {
    PolyPair& p = *ii;
    if (p.useless())
      continue;
    if (!h->head()->divides(*p.lcm, localRing))
      continue;
    lcm_t.makeLcm(*h->head(), *p.a->head(), localRing);
    if (lcm_t.equals(*p.lcm, localRing))
      continue;
    lcm_t.makeLcm(*h->head(), *p.b->head(), localRing);
    if (lcm_t.equals(*p.lcm, localRing))
      continue;
    p.makeUseless();
    bkUpdate += 1;
  }
  
  // Successive application of various deletion criteria
  Term& lcm_hi = localRing.makeTerm();
  Term& lcm_hj = localRing.makeTerm();
  for (PolySet::const_iterator ii = basis.begin(), ei = basis.end(); ii != ei; ++ii) {

    // Buchberger's Product criterion 
    if (h->head()->relPrime(*(*ii)->head(), localRing)) {
      bpUpdate += 1;
      continue;
    }

    // Gebauer-Moeller criteria M and F
    lcm_hi.makeLcm(*h->head(), *(*ii)->head(), localRing);
    bool condM = false;
    for (PolySet::const_iterator jj = ii + 1, ej = basis.end(); jj != ej; ++jj) {
      lcm_hj.makeLcm(*h->head(), *(*jj)->head(), localRing);
      if (lcm_hj.divides(lcm_hi, localRing)) {
        condM = true;
        break;
      }
    }

    if (!condM) {
      Term& lcm = ring.makeTerm(lcm_hi);
      out.push(&pairs.push(PolyPair(h, *ii, &lcm, ring.rankPair(*h, **ii))));
    } else {
      mfUpdate += 1;
    }
  }

  basis.push(h);
}

Galois::Statistic zeroUpdate("ZeroUpdate");

template<class R>
struct Process {
  typedef typename R::template realloc<Galois::PerIterAllocTy::rebind<char>::other>::other LocalRing;

  PolySet& basis;
  Galois::InsertBag<PolyPair>& pairs;
  R& ring;

  Process(PolySet& _basis, Galois::InsertBag<PolyPair>& _pairs, R& r): basis(_basis), pairs(_pairs), ring(r) { }

  void operator()(const PolyPair* p, Galois::UserContext<PolyPair*>& ctx) {
    if (p->useless()) {
      return;
    }

    LocalRing localRing(ring.numVars(), ctx.getPerIterAlloc());

    Poly* s = spoly(*p->lcm, *p->a, *p->b, localRing);
    Poly* h = reduce(*s, basis, localRing);

    if (!h->empty()) {
      Poly* hh = &ring.makePoly(*h);
      update(hh, basis, pairs, localRing, ring, ctx);
    } else {
      zeroUpdate += 1;
    }
  }
};

template<class R>
struct Verifier {
  PolySet& g;
  R& ring;

  Verifier(PolySet& _g, R& r): g(_g), ring(r) { }

  bool operator()(const PolyPair& p) {
    // TODO opportunity for temporary
    Poly* s = spoly(*p.lcm, *p.a, *p.b, ring);
    Poly* h = reduce(*s, g, ring);
    return !h->empty();
  }
};

template<class C,class R>
void allPairs(const PolySet& ideal, C& c, R& ring) {
  for (PolySet::const_iterator ii = ideal.begin(), ei = ideal.end(); ii != ei; ++ii) {
    for (PolySet::const_iterator jj = ideal.begin(), ej = ideal.end(); jj != ej; ++jj) {
      if (*ii == *jj)
        continue;
      if ((*ii)->empty() || (*jj)->empty())
        continue;
      Term& lcm = ring.makeTerm();
      lcm.makeLcm(*(*ii)->head(), *(*jj)->head(), ring);
      c.push(PolyPair(*ii, *jj, &lcm, ring.rankPair(**ii, **jj)));
    }
  }
}

struct Indexer {
  int operator()(const PolyPair& p) const {
    return p.index;
  }
  int operator()(const PolyPair* p) const {
    return p->index;
  }
};

template<class R>
void interReduce(PolySet& polys, R& ring) {
  for (PolySet::iterator ii = polys.begin(), ei = polys.end(); ii != ei; ++ii) {
    Poly*& p = *ii;
    Poly* reduced = reduce(*p, polys, ring);
    // TODO opportunity for temporary
    boost::swap(p, reduced);
    if (p->empty())
      continue;
    Field::element_type s(p->head()->coef());
    TheField.inverse(s);
    p->scaleBy(s, ring);
  }
}

template<class R>
void buchberger(PolySet& ideal, PolySet& basis, R& ring) {
  Galois::InsertBag<PolyPair> pairs;
  Galois::InsertBag<PolyPair*> initial;

  for (PolySet::iterator ii = ideal.begin(), ei = ideal.end(); ii != ei; ++ii) {
    if ((*ii)->empty())
      continue;
    update(*ii, basis, pairs, ring, ring, initial);
  }
  using namespace Galois::WorkList;
  typedef OrderedByIntegerMetric<Indexer,dChunkedLIFO<8> > OBIM;
  Galois::for_each(initial.begin(), initial.end(), Process<R>(basis, pairs, ring), Galois::wl<OBIM>());

  interReduce(basis, ring);
}

namespace parser {
namespace qi = boost::spirit::qi;
namespace ascii = boost::spirit::ascii;
namespace fusion = boost::fusion;

struct Coef {
  typedef boost::variant<char,char> Sign;
  Sign sign;
  typedef boost::variant<fusion::vector<int,int>, int> Rational;
  Galois::optional<Rational> rational;
};

struct Mono {
  std::string id;
  Galois::optional<int> expo;
};

struct Term {
  Coef coef;
  std::vector<Mono> monos;
};

struct Poly {
  std::vector<Term> terms;
};
}

// Macros need to be used outside of namespace scope

BOOST_FUSION_ADAPT_STRUCT(
    parser::Coef,
    (parser::Coef::Sign, sign)
    (Galois::optional<parser::Coef::Rational>, rational)
)

BOOST_FUSION_ADAPT_STRUCT(
    parser::Mono,
    (std::string, id)
    (Galois::optional<int>, expo)
)

BOOST_FUSION_ADAPT_STRUCT(
    parser::Term,
    (parser::Coef, coef)
    (std::vector<parser::Mono>, monos)
)

BOOST_FUSION_ADAPT_STRUCT(
    parser::Poly,
    (std::vector<parser::Term>, terms)
)

namespace parser {
namespace qi = boost::spirit::qi;
namespace ascii = boost::spirit::ascii;
namespace fusion = boost::fusion;

// +2xy^3 - z^3, ...
template<typename It>
struct PolySetGrammar: qi::grammar<It, std::vector<parser::Poly>(), ascii::space_type> {
  qi::rule<It, std::string(), ascii::space_type> id;
  qi::rule<It, parser::Coef::Sign(), ascii::space_type> sign;
  qi::rule<It, parser::Coef::Rational(), ascii::space_type> rational;
  qi::rule<It, parser::Coef(), ascii::space_type> coef;
  qi::rule<It, int(), ascii::space_type> expo;
  qi::rule<It, parser::Mono(), ascii::space_type> mono;
  qi::rule<It, parser::Term(), ascii::space_type> term;
  qi::rule<It, std::vector<parser::Term>(), ascii::space_type> poly;
  qi::rule<It, std::vector<parser::Poly>(), ascii::space_type> start;

  PolySetGrammar(): PolySetGrammar::base_type(start, "polynomial list") {
    // sign := + | -
    sign %= qi::char_("+") | qi::char_("-");
    sign.name("sign");
    // rational := (int / int) | int
    rational %= (qi::int_ >> qi::lit("/") > qi::int_) | qi::int_;
    rational.name("rational");
    // coef := sign rational?
    coef %= sign >> -rational;
    coef.name("coefficent");
    // expo := ^ int
    expo %= qi::lit("^") > qi::int_;
    expo.name("exponent");
    // mono := id expo?
    id %= qi::lexeme[+ascii::alnum];
    id.name("identifier");
    mono %= id >> -expo;
    mono.name("mononomial");
    // term := coef "*"? (mono "*"?)*
    term %= coef > -qi::lit("*") >> *(mono >> -qi::lit("*"));
    term.name("term");
    // poly := term+
    poly %= +term;
    poly.name("polynomial");
    // start := poly (, poly)*
    start %= poly % ',';

    using boost::phoenix::val;
    using boost::phoenix::construct;
    qi::on_error<qi::fail>(start,
        std::cerr 
        << val("Error! Expecting ")
        << qi::labels::_4
        << val(" here: \"")
        << construct<std::string>(qi::labels::_3, qi::labels::_2)
        << val("\"\n")
    );
  }
};

// Q[x y z w ...]
template<class It>
struct HeaderGrammar: qi::grammar<It, std::vector<std::string>(), ascii::space_type> {
  qi::rule<It, std::string(), ascii::space_type> id;
  qi::rule<It, std::vector<std::string>(), ascii::space_type> start;

  HeaderGrammar(): HeaderGrammar::base_type(start, "Ring Definition") { 
    id %= qi::lexeme[+ascii::alnum];
    id.name("id");
    // Q[ <id> (, <id>)* ]
    start %=  qi::lit("Q[") >> id % ',' >> qi::lit("]");

    using boost::phoenix::val;
    using boost::phoenix::construct;
    qi::on_error<qi::fail>(start,
        std::cerr 
        << val("Error! Expecting ")
        << qi::labels::_4
        << val(" here: \"")
        << construct<std::string>(qi::labels::_3, qi::labels::_2)
        << val("\"\n")
    );
  }
};

//! A simple parser.
template<class Order>
class Parser: private boost::noncopyable {
  typedef ::Term TheTerm;
  typedef ::Poly ThePoly;
  typedef Ring<Order> TheRing;
  typedef std::map<std::string,int> NameMap;
  typedef std::vector<std::string> IdMap;
  NameMap nameMap;
  IdMap idMap;
  PolySet m_polys;
  Order order;
  TheRing* m_ring;

  template<class It>
  void readHeader(It begin, It end) {
    HeaderGrammar<It> grammar;
    bool r = qi::phrase_parse(begin, end, grammar, ascii::space, idMap);

    if (!r || begin != end) {
      std::cerr << "Parse failure.\n";
      abort();
    }

    // Parse header
    for (IdMap::iterator ii = idMap.begin(), ei = idMap.end(); ii != ei; ++ii) {
      if (nameMap.find(*ii) != nameMap.end()) {
        std::cerr << "Duplicate variable name: " << *ii << "\n";
        abort();
      }
      int index = nameMap.size();
      nameMap[*ii] = index;
    }
  }

  template<class It>
  void readPolys(It begin, It end, std::vector<parser::Poly>& polys) {
    PolySetGrammar<It> grammar;
    bool r = qi::phrase_parse(begin, end, grammar, ascii::space, polys);
    if (!r || begin != end) {
      std::cerr << "Parse failure at: " << std::string(begin, end) << "\n";
      abort();
    }
  }

  struct RationalVisitor: public boost::static_visitor<Field::element_type> {
    Field::element_type operator()(const fusion::vector<int,int>& x) const {
      Field::element_type a(fusion::at_c<0>(x));
      Field::element_type b(fusion::at_c<1>(x));
      TheField.divide(a, b);
      return a;
    }
    Field::element_type operator()(const int& x) const {
      return Field::element_type(x);
    }
  };

  void parseCoef(const Coef& coef, Field::element_type& c) {
    const char* sign = boost::get<char>(&coef.sign);
    bool neg = *sign == '-';

    TheField.assign(c, 1);
    if (coef.rational) {
      c = boost::apply_visitor(RationalVisitor(), *coef.rational);
    }
    //c.canonicalize();
    if (neg)
      TheField.negate(c);
  }

  void parseMono(const Mono& mono, TheTerm& t) {
    NameMap::iterator ii = nameMap.find(mono.id);
    if (ii == nameMap.end()) {
      std::cerr << "Unknown variable name: " << mono.id << "\n";
      abort();
    }
    int index = ii->second;
    int expo = mono.expo ? *mono.expo : 1;
    t.exp(index) += expo;
  }

  void parseTerm(const Term& term, std::vector<TheTerm*>& terms) {
    TheTerm& t = m_ring->makeTerm();
    std::for_each(term.monos.begin(), term.monos.end(), boost::bind(&Parser::parseMono, this, _1, boost::ref(t)));
    parseCoef(term.coef, t.coef());
    m_ring->generateTotalDegree(t);
    terms.push_back(&t);
  }

  struct GreaterThan {
    Ring<Order>& ring;
    GreaterThan(Ring<Order>& r): ring(r) { }
    bool operator()(const TheTerm* a, const TheTerm* b) const {
      return ring.gt(*a, *b);
    }
  };

  void parsePoly(const Poly& poly) {
    std::vector<TheTerm*> terms;
    std::for_each(poly.terms.begin(), poly.terms.end(), boost::bind(&Parser::parseTerm, this, _1, boost::ref(terms)));

    std::sort(terms.begin(), terms.end(), GreaterThan(*m_ring));
    ThePoly& p = m_ring->makePoly();
    for (std::vector<TheTerm*>::iterator ii = terms.begin(), ei = terms.end(); ii != ei; ++ii) {
      p.push(*ii);
    }
    m_polys.push(&p);
  }

  void writeHeader(std::ostream& out) const {
    out << "Q[";
    for (IdMap::const_iterator ii = idMap.begin(), ei = idMap.end(); ii != ei; ++ii) {
      out << *ii;
      if (ii + 1 != ei)
        out << ", ";
    }
    out << "]\n";
  }

  void writePolys(std::ostream& out, const PolySet& polys) const {
    m_ring->write(out, polys, &idMap);
    out << "\n";
  }

public:
  typedef TheRing RingTy;

  Parser(): m_ring(0) { }

  ~Parser() {
    if (m_ring) delete m_ring;
  }

  void read(std::istream& in) {
    std::string header;
    getline(in, header);
    readHeader(header.begin(), header.end());

    m_ring = new Ring<Order>(idMap.size());

    in.unsetf(std::ios::skipws);
    boost::spirit::istream_iterator begin(in), end;

    std::vector<Poly> polys;
    readPolys(begin, end, polys);
    std::for_each(polys.begin(), polys.end(), boost::bind(&Parser::parsePoly, this, _1));

    std::cout << "Rational ring of " << nameMap.size() << " (" << m_ring->numVars() << ") variables\n";
    std::cout << "Ideal of " << polys.size() << " polynomials\n";
  }

  void write(std::ostream& out, const PolySet& polys) {
    writeHeader(out);
    writePolys(out, polys);
  }

  RingTy& ring() {
    return *m_ring;
  }

  PolySet& polys() {
    return m_polys;
  }
};

}

template<class Order>
void run() {
  std::ifstream scanner(filename.c_str());
  if (!scanner.good()) {
    std::cerr << "Couldn't open file: " << filename << "\n";
    abort();
  }

  typedef parser::Parser<Order> ParserTy;
  ParserTy P;
  P.read(scanner);
  scanner.close();

  P.write(std::cout, P.polys()); // REMOVe
  
  TheField.init();

  Galois::StatTimer T;
  T.start();
  interReduce(P.polys(), P.ring());
  PolySet basis;
  buchberger(P.polys(), basis, P.ring());
  T.stop();
  
  if (!skipVerify) {
    Galois::InsertBag<PolyPair> pairs;
    allPairs(basis, pairs, P.ring());
    Verifier<typename ParserTy::RingTy> v(basis, P.ring());
    if (Galois::ParallelSTL::find_if(pairs.begin(), pairs.end(), v) != pairs.end()) {
      std::cerr << "Basis is not Groebner.\n";
      assert(0 && "Triangulation failed");
      abort();
    }
  }
  P.write(std::cout, basis);
  std::cout << "Groebner basis with " << std::distance(basis.begin(), basis.end()) << " polynomials\n";
}


int main(int argc, char** argv) {
  Galois::StatManager statManager;
  statManager.push(zeroUpdate);
  statManager.push(bkUpdate);
  statManager.push(mfUpdate);
  statManager.push(bpUpdate);

  LonestarStart(argc, argv, name, desc, url);

  for (unsigned i = 0; i < sizeof(powersOfTwo)/sizeof(*powersOfTwo); ++i)
    powersOfTwo[i] = 2 << (i - 1);

  switch (monomialOrder) {
    case lex: run<LexOrder>(); break;
    case grevlex: run<GrevlexOrder>(); break;
    default: abort();
  }

  return 0;
}
