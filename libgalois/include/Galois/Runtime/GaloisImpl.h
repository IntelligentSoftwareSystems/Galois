#include "Galois/Runtime/Executor_DoAll.h"

namespace Galois {
namespace Runtime {

template<typename RangeTy, typename FunctionTy, typename TupleTy>
void do_all_gen(const RangeTy& r, const FunctionTy& fn, const TupleTy& tpl) {
  using Galois::loopname_tag;
  using Galois::loopname;

  static_assert(!exists_by_supertype<char*, TupleTy>::value, "old loopname");
  static_assert(!exists_by_supertype<char const *, TupleTy>::value, "old loopname");
  static_assert(!exists_by_supertype<bool, TupleTy>::value, "old steal");

  do_all_impl(
      r, fn,
      get_by_supertype_or_default<loopname_tag>(tpl, loopname{"(NULL)"}).getValue(),
      get_by_supertype_or_default<do_all_steal_tag>(tpl, do_all_steal<false>()).getValue());
}

template<typename RangeTy, typename FunctionTy, typename... Args>
  void for_each_gen(const RangeTy& r, const FunctionTy& fn, std::tuple<Args...> args);

template<typename FunctionTy, typename... Args>
void on_each_gen(const FunctionTy& fn, std::tuple<Args...> args);

} //namespace Runtime
} //namespace Galois
