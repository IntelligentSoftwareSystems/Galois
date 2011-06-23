// galois type traits -*- C++ -*-
/*
Galois, a framework to exploit amorphous data-parallelism in irregular
programs.

Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS SOFTWARE
AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY, FITNESS FOR ANY
PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF PERFORMANCE, AND ANY
WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF DEALING OR USAGE OF TRADE.
NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH RESPECT TO THE USE OF THE
SOFTWARE OR DOCUMENTATION. Under no circumstances shall University be liable
for incidental, special, indirect, direct or consequential damages or loss of
profits, interruption of business, or related expenses which may arise from use
of Software or Documentation, including but not limited to those resulting from
defects in Software and/or Documentation, or loss or inaccuracy of data of any
kind.
*/

namespace Galois {

//needs_parallel_pause indicates the operator may request the parallel loop to be suspended and a given function run in serial

BOOST_MPL_HAS_XXX_TRAIT_DEF(tt_needs_parallel_pause)

template<typename T>
struct needs_parallel_pause : public has_tt_needs_parallel_pause<T> {};


//does_not_need_parallel_push indicates the operator may generate new work and push it on the worklist

BOOST_MPL_HAS_XXX_TRAIT_DEF(tt_does_not_need_parallel_push)

template<typename T>
struct does_not_need_parallel_push : public has_tt_does_not_need_parallel_push<T> {};

//needs_per_iter_mem indicates the operator may request the access to a per-iteration parallel memory pool

BOOST_MPL_HAS_XXX_TRAIT_DEF(tt_needs_per_iter_mem)

template<typename T>
struct needs_per_iter_mem : public has_tt_needs_per_iter_mem<T> {};

}

//Two ways to declare a typetrait:
//First, with a typedef or other valid name declaration
//struct process {
//  typedef int tt_needs_parallel_pause;
//  ....
//};
//Second way, specialize the function
//namespace Galois {
//template<>
//struct needs_parallel_pause<process> : public boost::true_type {};
//}
