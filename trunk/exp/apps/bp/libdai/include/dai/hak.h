/*  This file is part of libDAI - http://www.libdai.org/
 *
 *  Copyright (c) 2006-2011, The libDAI authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
 */


/// \file
/// \brief Defines class HAK, which implements a variant of Generalized Belief Propagation.
/// \idea Implement more general region graphs and corresponding Generalized Belief Propagation updates as described in [\ref YFW05].
/// \todo Use ClusterGraph instead of a vector<VarSet> for speed.
/// \todo Optimize this code for large factor graphs.
/// \todo Implement GBP parent-child  algorithm.


#ifndef __defined_libdai_hak_h
#define __defined_libdai_hak_h


#include <string>
#include <dai/daialg.h>
#include <dai/regiongraph.h>
#include <dai/enum.h>
#include <dai/properties.h>


namespace dai {


/// Approximate inference algorithm: implementation of single-loop ("Generalized Belief Propagation") and double-loop algorithms by Heskes, Albers and Kappen [\ref HAK03]
class HAK : public DAIAlgRG {
    private:
        /// Outer region beliefs
        std::vector<Factor>                _Qa;
        /// Inner region beliefs
        std::vector<Factor>                _Qb;
        /// Messages from outer to inner regions
        std::vector<std::vector<Factor> >  _muab;
        /// Messages from inner to outer regions
        std::vector<std::vector<Factor> >  _muba;
        /// Maximum difference encountered so far
        Real _maxdiff;
        /// Number of iterations needed
        size_t _iters;

    public:
        /// Parameters for HAK
        struct Properties {
            /// Enumeration of possible cluster choices
            /** The following cluster choices are defined:
             *   - MIN minimal clusters, i.e., one outer region for each maximal factor
             *   - DELTA one outer region for each variable and its Markov blanket
             *   - LOOP one cluster for each loop of length at most \a Properties::loopdepth, and in addition one cluster for each maximal factor
             *   - BETHE Bethe approximation (one outer region for each maximal factor, inner regions are single variables)
             */
            DAI_ENUM(ClustersType,MIN,BETHE,DELTA,LOOP);

            /// Enumeration of possible message initializations
            DAI_ENUM(InitType,UNIFORM,RANDOM);

            /// Verbosity (amount of output sent to stderr)
            size_t verbose;

            /// Maximum number of iterations
            size_t maxiter;

            /// Maximum time (in seconds)
            double maxtime;

            /// Tolerance for convergence test
            Real tol;

            /// Damping constant (0.0 means no damping, 1.0 is maximum damping)
            Real damping;

            /// How to choose the outer regions
            ClustersType clusters;

            /// How to initialize the messages
            InitType init;

            /// Use single-loop (GBP) or double-loop (HAK)
            bool doubleloop;

            /// Depth of loops (only relevant for \a clusters == \c ClustersType::LOOP)
            size_t loopdepth;
        } props;

    public:
    /// \name Constructors/destructors
    //@{
        /// Default constructor
        HAK() : DAIAlgRG(), _Qa(), _Qb(), _muab(), _muba(), _maxdiff(0.0), _iters(0U), props() {}

        /// Construct from FactorGraph \a fg and PropertySet \a opts
        /** \param fg Factor graph.
         *  \param opts Parameters @see Properties
         */
        HAK( const FactorGraph &fg, const PropertySet &opts );

        /// Construct from RegionGraph \a rg and PropertySet \a opts
        HAK( const RegionGraph &rg, const PropertySet &opts );
    //@}


    /// \name General InfAlg interface
    //@{
        virtual HAK* clone() const { return new HAK(*this); }
        virtual HAK* construct( const FactorGraph &fg, const PropertySet &opts ) const { return new HAK( fg, opts ); }
        virtual std::string name() const { return "HAK"; }
        virtual Factor belief( const VarSet &vs ) const;
        virtual std::vector<Factor> beliefs() const;
        virtual Real logZ() const;
        virtual void init();
        virtual void init( const VarSet &vs );
        virtual Real run();
        virtual Real maxDiff() const { return _maxdiff; }
        virtual size_t Iterations() const { return _iters; }
        virtual void setMaxIter( size_t maxiter ) { props.maxiter = maxiter; }
        virtual void setProperties( const PropertySet &opts );
        virtual PropertySet getProperties() const;
        virtual std::string printProperties() const;
    //@}


    /// \name Additional interface specific for HAK
    //@{
        /// Returns reference to message from outer region \a alpha to its \a _beta 'th neighboring inner region
        Factor & muab( size_t alpha, size_t _beta ) { return _muab[alpha][_beta]; }
        /// Returns reference to message the \a _beta 'th neighboring inner region of outer region \a alpha to that outer region
        Factor & muba( size_t alpha, size_t _beta ) { return _muba[alpha][_beta]; }
        /// Returns belief of outer region \a alpha
        const Factor& Qa( size_t alpha ) const { return _Qa[alpha]; };
        /// Returns belief of inner region \a beta
        const Factor& Qb( size_t beta ) const { return _Qb[beta]; };

        /// Runs single-loop algorithm (algorithm 1 in [\ref HAK03])
        Real doGBP();
        /// Runs double-loop algorithm (as described in section 4.2 of [\ref HAK03]), which always convergences
        Real doDoubleLoop();
    //@}

    private:
        /// Helper function for constructors
        void construct();
        /// Recursive procedure for finding clusters of variables containing loops of length at most \a length
        /** \param fg the factor graph
         *  \param allcl the clusters found so far
         *  \param newcl partial candidate cluster
         *  \param root start (and end) point of the loop
         *  \param length number of variables that may be added to \a newcl
         *  \param vars neighboring variables of \a newcl
         *  \return allcl all clusters of variables with loops of length at most \a length passing through root
         */
        void findLoopClusters( const FactorGraph &fg, std::set<VarSet> &allcl, VarSet newcl, const Var & root, size_t length, VarSet vars );
};


} // end of namespace dai


#endif
