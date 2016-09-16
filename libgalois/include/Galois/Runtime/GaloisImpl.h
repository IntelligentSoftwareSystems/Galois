#include "Galois/Runtime/Executor_DoAll.h"

namespace Galois {
namespace Runtime {

template<typename RangeTy, typename FunctionTy, typename TupleTy>
void do_all_gen(const RangeTy& r, const FunctionTy& fn, const TupleTy& tpl) {
  static_assert(!exists_by_supertype<char*, TupleTy>::value, "old loopname");
  static_assert(!exists_by_supertype<char const *, TupleTy>::value, "old loopname");
  static_assert(!exists_by_supertype<bool, TupleTy>::value, "old steal");

  auto dtpl = std::tuple_cat(tpl,
      get_default_trait_values(tpl,
        std::make_tuple(loopname_tag{}, do_all_steal_tag{}),
        std::make_tuple(loopname{}, do_all_steal<>{})));

  do_all_impl(
      r, fn,
      get_by_supertype<loopname_tag>(dtpl).getValue(),
      get_by_supertype<do_all_steal_tag>(dtpl).getValue());
}

template<typename FunctionTy, typename... Args>
on_each_gen(const FunctionTy& fnm std::tuple<Args...> args);

} //namespace Runtime
} //namespace Galois
