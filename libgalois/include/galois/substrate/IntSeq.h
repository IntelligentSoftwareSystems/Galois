namespace galois {
  namespace substrate {

    template<class T, T... I>
    struct integer_sequence
    {
      typedef T value_type;

      static constexpr std::size_t size() noexcept;
    };

    namespace internal {
      template<class T, T N, T Z, T ...S> struct gens : gens<T, N-1, Z, N-1, S...> {};
      template<class T, T Z, T ...S> struct gens<T, Z, Z, S...> {
	typedef integer_sequence<T, S...> type;
      };
    }

    template<std::size_t... I>
    using index_sequence = integer_sequence<std::size_t, I...>;

    template<class T, T N>
    using make_integer_sequence = typename internal::gens<T, N, std::integral_constant<T, 0>::value>::type;
    template<std::size_t N>
    using make_index_sequence = make_integer_sequence<std::size_t, N>;

    template<class... T>
    using index_sequence_for = make_index_sequence<sizeof...(T)>;

  }
}
