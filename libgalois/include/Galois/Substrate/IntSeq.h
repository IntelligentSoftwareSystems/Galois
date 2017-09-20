/** Simple Integer Sequence -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galoisis a framework to exploit
 * amorphous data-parallelism in irregular programs.
 *
 * Galois is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, version 2.1 of the
 * License.
 *
 * Galois is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with Galois.  If not, see
 * <http://www.gnu.org/licenses/>.
 *
 * @section Copyright
 *
 * Copyright (C) 2015, The University of Texas at Austin. All rights
 * reserved.
 *
 * @section Description
 *
 * Like standard integer_sequence, use until we depend on C++14
 *
 * @author Andrew Lenharth <andrew@lenharth.org>
 */

namespace galois {
  namespace Substrate {

    template<class T, T... I>
    struct integer_sequence
    {
      typedef T value_type;
      
      static constexpr std::size_t size() noexcept;
    };

    namespace detail {
      template<class T, T N, T Z, T ...S> struct gens : gens<T, N-1, Z, N-1, S...> {};
      template<class T, T Z, T ...S> struct gens<T, Z, Z, S...> {
	typedef integer_sequence<T, S...> type;
      };
    }
	
    template<std::size_t... I>
    using index_sequence = integer_sequence<std::size_t, I...>;
    
    template<class T, T N>
    using make_integer_sequence = typename detail::gens<T, N, std::integral_constant<T, 0>::value>::type;
    template<std::size_t N>
    using make_index_sequence = make_integer_sequence<std::size_t, N>;
    
    template<class... T>
    using index_sequence_for = make_index_sequence<sizeof...(T)>;
    
  }
}
