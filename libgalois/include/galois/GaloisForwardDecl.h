namespace galois {

template <typename RangeFunc, typename FunctionTy, typename... Args>
void for_each(const RangeFunc& rangeMaker, const FunctionTy& fn, const Args&... args);

template <typename RangeFunc, typename FunctionTy, typename... Args>
void do_all(const RangeFunc& rangeMaker, const FunctionTy& fn, const Args&... args);

template<typename FunctionTy, typename... Args>
void on_each(const FunctionTy& fn, const Args&... args);

template<typename FunctionTy, typename... Args>
void on_each(FunctionTy& fn, const Args&... args);

} // end namespace galois

