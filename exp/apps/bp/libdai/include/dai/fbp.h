/*  This file is part of libDAI - http://www.libdai.org/
 *
 *  Copyright (c) 2006-2011, The libDAI authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
 */


/// \file
/// \brief Defines class FBP, which implements Fractional Belief Propagation


#ifndef __defined_libdai_fbp_h
#define __defined_libdai_fbp_h


#include <string>
#include <dai/daialg.h>
#include <dai/factorgraph.h>
#include <dai/properties.h>
#include <dai/enum.h>
#include <dai/bp.h>


namespace dai {


/// Approximate inference algorithm "Fractional Belief Propagation" [\ref WiH03]
/** The Fractional Belief Propagation algorithm is like Belief
 *  Propagation, but associates each factor with a weight (scale parameter)
 *  which controls the divergence measure being minimized. Standard
 *  Belief Propagation corresponds to the case of FBP where each weight
 *  is 1. When cast as an EP algorithm, BP (and EP) minimize
 *  the inclusive KL-divergence, i.e. \f$\min_q KL(p||q)\f$ (note that the
 *  Bethe free energy is typically derived from \f$ KL(q||p) \f$). If each
 *  factor \a I has weight \f$ c_I \f$, then FBP minimizes the
 *  alpha-divergence with \f$ \alpha=1/c_I \f$ for that factor, which also
 *  corresponds to Power EP [\ref Min05].
 *
 *  The messages \f$m_{I\to i}(x_i)\f$ are passed from factors \f$I\f$ to variables \f$i\f$. 
 *  The update equation is given by:
 *    \f[ m_{I\to i}(x_i) \propto \left( \sum_{x_{N_I\setminus\{i\}}} f_I(x_I)^{1/c_I} \prod_{j\in N_I\setminus\{i\}} m_{I\to j}^{1-1/c_I} \prod_{J\in N_j\setminus\{I\}} m_{J\to j} \right)^{c_I} \f]
 *  After convergence, the variable beliefs are calculated by:
 *    \f[ b_i(x_i) \propto \prod_{I\in N_i} m_{I\to i} \f]
 *  and the factor beliefs are calculated by:
 *    \f[ b_I(x_I) \propto f_I(x_I)^{1/c_I} \prod_{j \in N_I} m_{I\to j}^{1-1/c_I} \prod_{J\in N_j\setminus\{I\}} m_{J\to j} \f]
 *  The logarithm of the partition sum is approximated by:
 *    \f[ \log Z = \sum_{I} \sum_{x_I} b_I(x_I) \big( \log f_I(x_I) - c_I \log b_I(x_I) \big) + \sum_{i} (c_i - 1) \sum_{x_i} b_i(x_i) \log b_i(x_i) \f]
 *  where the variable weights are defined as
 *    \f[ c_i := \sum_{I \in N_i} c_I \f]
 *
 *  \todo Add nice way to set weights
 *  \author Frederik Eaton
 */
class FBP : public BP {
    protected:
        /// Factor weights (indexed by factor ID)
        std::vector<Real> _weight;

    public:
    /// \name Constructors/destructors
    //@{
        /// Default constructor
        FBP() : BP(), _weight() {}

        /// Construct from FactorGraph \a fg and PropertySet \a opts
        /** \param fg Factor graph.
         *  \param opts Parameters @see BP::Properties
         */
        FBP( const FactorGraph &fg, const PropertySet &opts ) : BP(fg, opts), _weight() {
            setProperties( opts );
            construct();
        }
    //@}

    /// \name General InfAlg interface
    //@{
        virtual FBP* clone() const { return new FBP(*this); }
        virtual FBP* construct( const FactorGraph &fg, const PropertySet &opts ) const { return new FBP( fg, opts ); }
        virtual std::string name() const { return "FBP"; }
        virtual Real logZ() const;
    //@}

    /// \name FBP accessors/mutators for weights
    //@{
        /// Returns weight of the \a I 'th factor
        Real Weight( size_t I ) const { return _weight[I]; }

        /// Returns constant reference to vector of all factor weights
        const std::vector<Real>& Weights() const { return _weight; }

        /// Sets the weight of the \a I 'th factor to \a c
        void setWeight( size_t I, Real c ) { _weight[I] = c; }

        /// Sets the weights of all factors simultaenously
        /** \note Faster than calling setWeight(size_t,Real) for each factor
         */
        void setWeights( const std::vector<Real> &c ) { _weight = c; }

    protected:
        /// Calculate the product of factor \a I and the incoming messages
        /** If \a without_i == \c true, the message coming from variable \a i is omitted from the product
         *  \note This function is used by calcNewMessage() and calcBeliefF()
         */
        virtual Prob calcIncomingMessageProduct( size_t I, bool without_i, size_t i ) const;

        // Calculate the updated message from the \a _I 'th neighbor of variable \a i to variable \a i
        virtual void calcNewMessage( size_t i, size_t _I );

        // Calculates unnormalized belief of factor \a I
        virtual void calcBeliefF( size_t I, Prob &p ) const {
            p = calcIncomingMessageProduct( I, false, 0 );
        }

        // Helper function for constructors
        virtual void construct();
};


} // end of namespace dai


#endif
