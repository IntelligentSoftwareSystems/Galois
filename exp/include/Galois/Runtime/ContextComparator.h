#ifndef GALOIS_RUNTIME_CONTEXT_COMPARATOR_H
#define GALOIS_RUNTIME_CONTEXT_COMPARATOR_H

namespace Galois {
namespace Runtime {


// TODO: change comparator to three valued int instead of bool

template <typename Ctxt, typename Cmp>
struct ContextComparator {
  const Cmp& cmp;

  explicit ContextComparator (const Cmp& cmp): cmp (cmp) {}

  inline bool operator () (const Ctxt* left, const Ctxt* right) const {
    assert (left != NULL);
    assert (right != NULL);
    return cmp (left->getActive (), right->getActive ());
  }
};



} // end namespace Runtime
} // end namespace Galois


#endif // GALOIS_RUNTIME_CONTEXT_COMPARATOR_H
