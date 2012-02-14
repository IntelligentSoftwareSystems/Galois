/*  This file is part of libDAI - http://www.libdai.org/
 *
 *  Copyright (c) 2006-2011, The libDAI authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
 */


/// \file
/// \brief Defines class TreeEP, which implements Tree Expectation Propagation
/// \todo Clean up the TreeEP code (exploiting that a large part of the code
/// is just a special case of JTree).


#ifndef __defined_libdai_treeep_h
#define __defined_libdai_treeep_h


#include <vector>
#include <string>
#include <dai/daialg.h>
#include <dai/varset.h>
#include <dai/regiongraph.h>
#include <dai/factorgraph.h>
#include <dai/clustergraph.h>
#include <dai/weightedgraph.h>
#include <dai/jtree.h>
#include <dai/properties.h>
#include <dai/enum.h>


namespace dai {


/// Approximate inference algorithm "Tree Expectation Propagation" [\ref MiQ04]
class TreeEP : public JTree {
    private:
        /// Maximum difference encountered so far
        Real                  _maxdiff;
        /// Number of iterations needed
        size_t                _iters;

    public:
        /// Parameters for TreeEP
        struct Properties {
            /// Enumeration of possible choices for the tree
            /** The two possibilities are:
             *  - \c ORG: take the maximum spanning tree where the weights are crude
             *            estimates of the mutual information between the nodes;
             *  - \c ALT: take the maximum spanning tree where the weights are upper
             *            bounds on the effective interaction strengths between pairs of nodes.
             */
            DAI_ENUM(TypeType,ORG,ALT);

            /// Verbosity (amount of output sent to stderr)
            size_t verbose;

            /// Maximum number of iterations
            size_t maxiter;

            /// Maximum time (in seconds)
            double maxtime;

            /// Tolerance for convergence test
            Real tol;

            /// How to choose the tree
            TypeType type;
        } props;

    private:
        /// Stores the data structures needed to efficiently update the approximation of an off-tree factor.
        /** The TreeEP object stores a TreeEPSubTree object for each off-tree factor.
         *  It stores the approximation of that off-tree factor, which is represented 
         *  as a distribution on a subtree of the main tree.
         */
        class TreeEPSubTree {
            private:
                /// Outer region pseudomarginals (corresponding with the \f$\tilde f_i(x_j,x_k)\f$ in [\ref MiQ04])
                std::vector<Factor>  _Qa;
                /// Inner region pseudomarginals (corresponding with the \f$\tilde f_i(x_s)\f$ in [\ref MiQ04])
                std::vector<Factor>  _Qb;
                /// The junction tree (stored as a rooted tree)
                RootedTree           _RTree;
                /// Index conversion table for outer region indices (_Qa[alpha] corresponds with Qa[_a[alpha]] of the supertree)
                std::vector<size_t>  _a;        
                /// Index conversion table for inner region indices (_Qb[beta] corresponds with Qb[_b[beta]] of the supertree)
                std::vector<size_t>  _b;
                /// Pointer to off-tree factor
                const Factor *       _I;
                /// Variables in off-tree factor
                VarSet               _ns;
                /// Variables in off-tree factor which are not in the root of this subtree
                VarSet               _nsrem;
                /// Used for calculating the free energy
                Real                 _logZ;

            public:
            /// \name Constructors/destructors
            //@{
                /// Default constructor
                TreeEPSubTree() : _Qa(), _Qb(), _RTree(), _a(), _b(), _I(NULL), _ns(), _nsrem(), _logZ(0.0) {}

                /// Copy constructor
                TreeEPSubTree( const TreeEPSubTree &x ) : _Qa(x._Qa), _Qb(x._Qb), _RTree(x._RTree), _a(x._a), _b(x._b), _I(x._I), _ns(x._ns), _nsrem(x._nsrem), _logZ(x._logZ) {}

                /// Assignment operator
                TreeEPSubTree & operator=( const TreeEPSubTree& x ) {
                    if( this != &x ) {
                        _Qa         = x._Qa;
                        _Qb         = x._Qb;
                        _RTree      = x._RTree;
                        _a          = x._a;
                        _b          = x._b;
                        _I          = x._I;
                        _ns         = x._ns;
                        _nsrem      = x._nsrem;
                        _logZ       = x._logZ;
                    }
                    return *this;
                }

                /// Construct from \a subRTree, which is a subtree of the main tree \a jt_RTree, with distribution represented by \a jt_Qa and \a jt_Qb, for off-tree factor \a I
                TreeEPSubTree( const RootedTree &subRTree, const RootedTree &jt_RTree, const std::vector<Factor> &jt_Qa, const std::vector<Factor> &jt_Qb, const Factor *I );
            //@}

                /// Initializes beliefs of this subtree
                void init();

                /// Inverts this approximation and multiplies it by the (super) junction tree marginals \a Qa and \a Qb
                void InvertAndMultiply( const std::vector<Factor> &Qa, const std::vector<Factor> &Qb );

                /// Runs junction tree algorithm (including off-tree factor I) storing the results in the (super) junction tree \a Qa and \a Qb
                void HUGIN_with_I( std::vector<Factor> &Qa, std::vector<Factor> &Qb );

                /// Returns energy (?) of this subtree
                Real logZ( const std::vector<Factor> &Qa, const std::vector<Factor> &Qb ) const;

                /// Returns constant reference to the pointer to the off-tree factor
                const Factor *& I() { return _I; }
        };

        /// Stores a TreeEPSubTree object for each off-tree factor
        std::map<size_t, TreeEPSubTree>  _Q;

    public:
        /// Default constructor
        TreeEP() : JTree(), _maxdiff(0.0), _iters(0), props(), _Q() {}

        /// Copy constructor
        TreeEP( const TreeEP &x ) : JTree(x), _maxdiff(x._maxdiff), _iters(x._iters), props(x.props), _Q(x._Q) {
            for( size_t I = 0; I < nrFactors(); I++ )
                if( offtree( I ) )
                    _Q[I].I() = &factor(I);
        }

        /// Assignment operator
        TreeEP& operator=( const TreeEP &x ) {
            if( this != &x ) {
                JTree::operator=( x );
                _maxdiff = x._maxdiff;
                _iters   = x._iters;
                props    = x.props;
                _Q       = x._Q;
                for( size_t I = 0; I < nrFactors(); I++ )
                    if( offtree( I ) )
                        _Q[I].I() = &factor(I);
            }
            return *this;
        }

        /// Construct from FactorGraph \a fg and PropertySet \a opts
        /** \param fg Factor graph.
         *  \param opts Parameters @see Properties
         */
        TreeEP( const FactorGraph &fg, const PropertySet &opts );


    /// \name General InfAlg interface
    //@{
        virtual TreeEP* clone() const { return new TreeEP(*this); }
        virtual TreeEP* construct( const FactorGraph &fg, const PropertySet &opts ) const { return new TreeEP( fg, opts ); }
        virtual std::string name() const { return "TREEEP"; }
        virtual Real logZ() const;
        virtual void init();
        virtual void init( const VarSet &/*ns*/ ) { init(); }
        virtual Real run();
        virtual Real maxDiff() const { return _maxdiff; }
        virtual size_t Iterations() const { return _iters; }
        virtual void setMaxIter( size_t maxiter ) { props.maxiter = maxiter; }
        virtual void setProperties( const PropertySet &opts );
        virtual PropertySet getProperties() const;
        virtual std::string printProperties() const;
    //@}

    private:
        /// Helper function for constructors
        void construct( const FactorGraph& fg, const RootedTree& tree );
        /// Returns \c true if factor \a I is not part of the tree
        bool offtree( size_t I ) const { return (fac2OR(I) == -1U); }
};


} // end of namespace dai


#endif
