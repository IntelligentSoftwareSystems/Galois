/*  This file is part of libDAI - http://www.libdai.org/
 *
 *  Copyright (c) 2006-2011, The libDAI authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
 */


/// \file
/// \brief Defines class MR, which implements loop corrections as proposed by Montanari and Rizzo


#ifndef __defined_libdai_mr_h
#define __defined_libdai_mr_h


#include <vector>
#include <string>
#include <dai/factorgraph.h>
#include <dai/daialg.h>
#include <dai/enum.h>
#include <dai/properties.h>
#include <dai/exceptions.h>
#include <dai/graph.h>
#include <boost/dynamic_bitset.hpp>


namespace dai {


/// Approximate inference algorithm by Montanari and Rizzo [\ref MoR05]
/** \author Bastian Wemmenhove wrote the original implementation before it was merged into libDAI
 */
class MR : public DAIAlgFG {
    private:
        /// Is the underlying factor graph supported?
        bool supported;

        /// The interaction graph (Markov graph)
        GraphAL G;

        /// tJ[i][_j] is the hyperbolic tangent of the interaction between spin \a i and its neighbour G.nb(i,_j)
        std::vector<std::vector<Real> >                 tJ;
        /// theta[i] is the local field on spin \a i
        std::vector<Real>                               theta;

        /// M[i][_j] is \f$ M^{(i)}_j \f$
        std::vector<std::vector<Real> >                 M;
        /// Cavity correlations
        std::vector<std::vector<std::vector<Real> > >   cors;

        /// Type used for managing a subset of neighbors
        typedef boost::dynamic_bitset<> sub_nb;
        
        /// Magnetizations
        std::vector<Real> Mag;
        
        /// Maximum difference encountered so far
        Real _maxdiff;
        
        /// Number of iterations needed
        size_t _iters;

    public:
        /// Parameters for MR
        struct Properties {
            /// Enumeration of different types of update equations
            /** The possible update equations are:
             *  - FULL full updates, slow but accurate
             *  - LINEAR linearized updates, faster but less accurate
             */
            DAI_ENUM(UpdateType,FULL,LINEAR);

            /// Enumeration of different ways of initializing the cavity correlations
            /** The possible cavity initializations are:
             *  - RESPPROP using response propagation ("linear response")
             *  - CLAMPING using clamping and BP
             *  - EXACT using JunctionTree
             */
            DAI_ENUM(InitType,RESPPROP,CLAMPING,EXACT);

            /// Verbosity (amount of output sent to stderr)
            size_t verbose;

            /// Tolerance for convergence test
            Real tol;

            /// Update equations
            UpdateType updates;

            /// How to initialize the cavity correlations
            InitType inits;
        } props;

    public:
        /// Default constructor
        MR() : DAIAlgFG(), supported(), G(), tJ(), theta(), M(), cors(), Mag(), _maxdiff(), _iters(), props() {}

        /// Construct from FactorGraph \a fg and PropertySet \a opts
        /** \param fg Factor graph.
         *  \param opts Parameters @see Properties
         *  \note This implementation only deals with binary variables and pairwise interactions.
         *  \throw NOT_IMPLEMENTED if \a fg has factors depending on three or more variables or has variables with more than two possible states.
         */
        MR( const FactorGraph &fg, const PropertySet &opts );


    /// \name General InfAlg interface
    //@{
        virtual MR* clone() const { return new MR(*this); }
        virtual MR* construct( const FactorGraph &fg, const PropertySet &opts ) const { return new MR( fg, opts ); }
        virtual std::string name() const { return "MR"; }
        virtual Factor belief( const Var &v ) const { return beliefV( findVar( v ) ); }
        virtual Factor belief( const VarSet &/*vs*/ ) const;
        virtual Factor beliefV( size_t i ) const;
        virtual std::vector<Factor> beliefs() const;
        virtual Real logZ() const { DAI_THROW(NOT_IMPLEMENTED); return 0.0; }
        virtual void init() {}
        virtual void init( const VarSet &/*ns*/ ) { DAI_THROW(NOT_IMPLEMENTED); }
        virtual Real run();
        virtual Real maxDiff() const { return _maxdiff; }
        virtual size_t Iterations() const { return _iters; }
        virtual void setProperties( const PropertySet &opts );
        virtual PropertySet getProperties() const;
        virtual std::string printProperties() const;
    //@}

    private:
        /// Initialize cors
        Real calcCavityCorrelations();
        
        /// Iterate update equations for cavity fields
        void propagateCavityFields();
        
        /// Calculate magnetizations
        void calcMagnetizations();

        /// Calculate the product of all tJ[i][_j] for _j in A
        /** \param i variable index
         *  \param A subset of neighbors of variable \a i
         */
        Real _tJ(size_t i, sub_nb A);
        
        /// Calculate \f$ \Omega^{(i)}_{j,l} \f$ as defined in [\ref MoR05] eqn. (2.15)
        Real Omega(size_t i, size_t _j, size_t _l);
        
        /// Calculate \f$ T^{(i)}_A \f$ as defined in [\ref MoR05] eqn. (2.17) with \f$ A = \{l_1,l_2,\dots\} \f$
        /** \param i variable index
         *  \param A subset of neighbors of variable \a i
         */
        Real T(size_t i, sub_nb A);
        
        /// Calculates \f$ T^{(i)}_j \f$ where \a j is the \a _j 'th neighbor of \a i
        Real T(size_t i, size_t _j);
        
        /// Calculates \f$ \Gamma^{(i)}_{j,l_1l_2} \f$ as defined in [\ref MoR05] eqn. (2.16)
        Real Gamma(size_t i, size_t _j, size_t _l1, size_t _l2);
        
        /// Calculates \f$ \Gamma^{(i)}_{l_1l_2} \f$ as defined in [\ref MoK07] on page 1141
        Real Gamma(size_t i, size_t _l1, size_t _l2);
        
        /// Approximates moments of variables in \a A
        /** Calculate the moment of variables in \a A from M and cors, neglecting higher order cumulants,
         *  defined as the sum over all partitions of A into subsets of cardinality two at most of the
         *  product of the cumulants (either first order, i.e. M, or second order, i.e. cors) of the
         *  entries of the partitions.
         *
         *  \param i variable index
         *  \param A subset of neighbors of variable \a i
         */
        Real appM(size_t i, sub_nb A);
        
        /// Calculate sum over all even/odd subsets B of \a A of _tJ(j,B) appM(j,B)
        /** \param j variable index
         *  \param A subset of neighbors of variable \a j
         *  \param sum_even on return, will contain the sum over all even subsets
         *  \param sum_odd on return, will contain the sum over all odd subsets
         */
        void sum_subs(size_t j, sub_nb A, Real *sum_even, Real *sum_odd);
};


} // end of namespace dai


#endif
