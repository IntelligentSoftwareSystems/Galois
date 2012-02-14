/*  This file is part of libDAI - http://www.libdai.org/
 *
 *  Copyright (c) 2006-2011, The libDAI authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
 */


/// \file
/// \brief Defines class MF which implements the Mean Field algorithm


#ifndef __defined_libdai_mf_h
#define __defined_libdai_mf_h


#include <string>
#include <dai/enum.h>
#include <dai/daialg.h>
#include <dai/factorgraph.h>
#include <dai/properties.h>


namespace dai {


/// Approximate inference algorithm "Mean Field"
/** The Mean Field algorithm iteratively calculates approximations of
 *  single variable marginals (beliefs). The update equation for 
 *  a single belief \f$b_i\f$ is given by:
 *    \f[ b_i^{\mathrm{new}}(x_i) \propto \prod_{I\in N_i} \exp \left( \sum_{x_{N_I \setminus \{i\}}} \log f_I(x_I) \prod_{j \in N_I \setminus \{i\}} b_j(x_j) \right) \f]
 *  for naive mean field and by
 *    \f[ b_i^{\mathrm{new}}(x_i) \propto \prod_{I\in N_i} \left( \sum_{x_{N_I \setminus \{i\}}} f_I(x_I) \prod_{j \in N_I \setminus \{i\}} b_j(x_j) \right) \f]
 *  for hard-spin mean field.
 *  These update equations are performed for all variables until convergence.
 */
class MF : public DAIAlgFG {
    private:
        /// Current approximations of single variable marginals
        std::vector<Factor>  _beliefs;
        /// Maximum difference encountered so far
        Real _maxdiff;
        /// Number of iterations needed
        size_t _iters;

    public:
        /// Parameters for MF
        struct Properties {
            /// Enumeration of possible message initializations
            DAI_ENUM(InitType,UNIFORM,RANDOM);

            /// Enumeration of possible update types
            DAI_ENUM(UpdateType,NAIVE,HARDSPIN);

            /// Verbosity (amount of output sent to stderr)
            size_t verbose;

            /// Maximum number of iterations
            size_t maxiter;

            /// Tolerance for convergence test
            Real tol;

            /// Damping constant (0.0 means no damping, 1.0 is maximum damping)
            Real damping;
            
            /// How to initialize the messages/beliefs
            InitType init;

            /// How to update the messages/beliefs
            UpdateType updates;
        } props;

    public:
    /// \name Constructors/destructors
    //@{
        /// Default constructor
        MF() : DAIAlgFG(), _beliefs(), _maxdiff(0.0), _iters(0U), props() {}

        /// Construct from FactorGraph \a fg and PropertySet \a opts
        /** \param fg Factor graph.
         *  \param opts Parameters @see Properties
         */
        MF( const FactorGraph &fg, const PropertySet &opts ) : DAIAlgFG(fg), _beliefs(), _maxdiff(0.0), _iters(0U), props() {
            setProperties( opts );
            construct();
        }
    //@}

    /// \name General InfAlg interface
    //@{
        virtual MF* clone() const { return new MF(*this); }
        virtual MF* construct( const FactorGraph &fg, const PropertySet &opts ) const { return new MF( fg, opts ); }
        virtual std::string name() const { return "MF"; }
        virtual Factor belief( const Var &v ) const { return beliefV( findVar( v ) ); }
        virtual Factor belief( const VarSet &vs ) const;
        virtual Factor beliefV( size_t i ) const;
        virtual std::vector<Factor> beliefs() const;
        virtual Real logZ() const;
        virtual void init();
        virtual void init( const VarSet &ns );
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
        void construct();

        /// Calculates an updated belief of variable \a i
        Factor calcNewBelief( size_t i );
};


} // end of namespace dai


#endif
