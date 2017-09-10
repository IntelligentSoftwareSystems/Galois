/*  This file is part of libDAI - http://www.libdai.org/
 *
 *  Copyright (c) 2006-2011, The libDAI authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
 */


/// \file
/// \brief Defines the VarSet class, which represents a set of random variables.


#ifndef __defined_libdai_varset_h
#define __defined_libdai_varset_h


#include <vector>
#include <map>
#include <ostream>
#include <dai/var.h>
#include <dai/util.h>
#include <dai/smallset.h>


namespace dai {


// Predefine for definitions of calcLinearState() and calcState()
class VarSet;


/// Calculates the linear index in the Cartesian product of the variables in \a vs that corresponds to a particular joint assignment of the variables, specified by \a state.
/** \param vs Set of variables for which the linear state should be calculated;
 *  \param state Specifies the states of some variables.
 *  \return The linear index in the Cartesian product of the variables in \a vs
 *  corresponding with the joint assignment specified by \a state, where variables
 *  for which no state is specified are assumed to be in state 0.
 *
 *  The linear index is calculated as follows. The variables in \a vs are
 *  ordered according to their label (in ascending order); say \a vs corresponds with
 *  the set \f$\{x_{l(0)},x_{l(1)},\dots,x_{l(n-1)}\}\f$ with \f$l(0) < l(1) < \dots < l(n-1)\f$,
 *  where variable \f$x_l\f$ has label \a l. Denote by \f$S_l\f$ the number of possible values
 *  ("states") of variable \f$x_l\f$. The argument \a state corresponds
 *  with a mapping \a s that assigns to each variable \f$x_l\f$ a state \f$s(x_l) \in \{0,1,\dots,S_l-1\}\f$,
 *  where \f$s(x_l)=0\f$ if \f$x_l\f$ is not specified in \a state. The linear index \f$S\f$ corresponding
 *  with \a state is now calculated by:
 *  \f{eqnarray*}
 *    S &:=& \sum_{i=0}^{n-1} s(x_{l(i)}) \prod_{j=0}^{i-1} S_{l(j)} \\
 *      &= & s(x_{l(0)}) + s(x_{l(1)}) S_{l(0)} + s(x_{l(2)}) S_{l(0)} S_{l(1)} + \dots + s(x_{l(n-1)}) S_{l(0)} \cdots S_{l(n-2)}.
 *  \f}
 *
 *  \note If \a vs corresponds with \f$\{x_l\}_{l\in L}\f$, and \a state specifies a state
 *  for each variable \f$x_l\f$ for \f$l\in L\f$, calcLinearState() induces a mapping
 *  \f$\sigma : \prod_{l\in L} X_l \to \{0,1,\dots,\prod_{l\in L} S_l-1\}\f$ that
 *  maps a joint state to a linear index; this is the inverse of the mapping
 *  \f$\sigma^{-1}\f$ induced by calcState().
 *
 *  \see calcState()
 */
size_t calcLinearState( const VarSet &vs, const std::map<Var, size_t> &state );


/// Calculates the joint assignment of the variables in \a vs corresponding to the linear index \a linearState.
/** \param vs Set of variables to which \a linearState refers
 *  \param linearState should be smaller than vs.nrStates().
 *  \return A mapping \f$s\f$ that maps each Var \f$x_l\f$ in \a vs to its state \f$s(x_l)\f$, as specified by \a linearState.
 *
 *  The variables in \a vs are ordered according to their label (in ascending order); say \a vs corresponds with
 *  the set \f$\{x_{l(0)},x_{l(1)},\dots,x_{l(n-1)}\}\f$ with \f$l(0) < l(1) < \dots < l(n-1)\f$,
 *  where variable \f$x_l\f$ has label \a l. Denote by \f$S_l\f$ the number of possible values
 *  ("states") of variable \f$x_l\f$ with label \a l.
 *  The mapping \f$s\f$ returned by this function is defined as:
 *  \f{eqnarray*}
 *    s(x_{l(i)}) = \left\lfloor\frac{S \mbox { mod } \prod_{j=0}^{i} S_{l(j)}}{\prod_{j=0}^{i-1} S_{l(j)}}\right\rfloor \qquad \mbox{for all $i=0,\dots,n-1$}.
 *  \f}
 *  where \f$S\f$ denotes the value of \a linearState.
 *
 *  \note If \a vs corresponds with \f$\{x_l\}_{l\in L}\f$, calcState() induces a mapping
 *  \f$\sigma^{-1} : \{0,1,\dots,\prod_{l\in L} S_l-1\} \to \prod_{l\in L} X_l\f$ that
 *  maps a linear index to a joint state; this is the inverse of the mapping \f$\sigma\f$
 *  induced by calcLinearState().
 *
 *  \see calcLinearState()
 */
std::map<Var, size_t> calcState( const VarSet &vs, size_t linearState );


/// Represents a set of variables.
/** \note A VarSet is implemented using a SmallSet<Var> instead
 *  of the more natural std::set<Var> because of efficiency reasons.
 *  That is, internally, the variables in the set are sorted ascendingly
 *  according to their labels.
 */
class VarSet : public SmallSet<Var> {
    public:
    /// \name Constructors and destructors
    //@{
        /// Default constructor (constructs an empty set)
        VarSet() : SmallSet<Var>() {}

        /// Construct from \link SmallSet \endlink<\link Var \endlink> \a x
        VarSet( const SmallSet<Var> &x ) : SmallSet<Var>(x) {}

        /// Construct a VarSet with one element, \a v
        VarSet( const Var &v ) : SmallSet<Var>(v) {}

        /// Construct a VarSet with two elements, \a v1 and \a v2
        VarSet( const Var &v1, const Var &v2 ) : SmallSet<Var>(v1,v2) {}

        /// Construct a VarSet from the range between \a begin and \a end.
        /** \tparam VarIterator Iterates over instances of type Var.
         *  \param begin Points to first Var to be added.
         *  \param end Points just beyond last Var to be added.
         *  \param sizeHint For efficiency, the number of elements can be speficied by \a sizeHint.
         */
        template <typename VarIterator>
        VarSet( VarIterator begin, VarIterator end, size_t sizeHint=0 ) : SmallSet<Var>(begin,end,sizeHint) {}
    //@}

    /// \name Queries
    //@{
        /// Calculates the number of states of this VarSet, which is simply the number of possible joint states of the variables in \c *this.
        /** The number of states of the Cartesian product of the variables in this VarSet
         *  is simply the product of the number of states of each variable in this VarSet.
         *  If \c *this corresponds with the set \f$\{x_l\}_{l\in L}\f$,
         *  where variable \f$x_l\f$ has label \f$l\f$, and denoting by \f$S_l\f$ the
         *  number of possible values ("states") of variable \f$x_l\f$, the number of
         *  joint configurations of the variables in \f$\{x_l\}_{l\in L}\f$ is given by \f$\prod_{l\in L} S_l\f$.
         */
        BigInt nrStates() const {
            BigInt states = 1;
            for( VarSet::const_iterator n = begin(); n != end(); n++ )
                states *= n->states();
            return states;
        }
    //@}

    /// \name Input and output
    //@{
        /// Writes a VarSet to an output stream
        friend std::ostream& operator<<( std::ostream &os, const VarSet &vs )  {
            os << "{";
            for( VarSet::const_iterator v = vs.begin(); v != vs.end(); v++ )
                os << (v != vs.begin() ? ", " : "") << *v;
            os << "}";
            return( os );
        }
    //@}
};


} // end of namespace dai


/** \example example_varset.cpp
 *  This example shows how to use the Var, VarSet and State classes. It also explains the concept of "states" for VarSets.
 *
 *  \section Output
 *  \verbinclude examples/example_varset.out
 *
 *  \section Source
 */


#endif
