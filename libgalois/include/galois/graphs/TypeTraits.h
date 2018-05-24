#ifndef GALOIS_GRAPH_TYPETRAITS_H
#define GALOIS_GRAPH_TYPETRAITS_H

#include <boost/mpl/has_xxx.hpp>

namespace galois {
namespace graphs {

BOOST_MPL_HAS_XXX_TRAIT_DEF(tt_is_segmented)
template<typename T>
struct is_segmented: public has_tt_is_segmented<T> {};

}
}
#endif
