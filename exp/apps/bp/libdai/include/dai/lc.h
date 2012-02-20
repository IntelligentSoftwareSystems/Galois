/*  This file is part of libDAI - http://www.libdai.org/
 *
 *  Copyright (c) 2006-2011, The libDAI authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
 */


/// \file
/// \brief Defines class LC, which implements loop corrections for approximate inference


#ifndef __defined_libdai_lc_h
#define __defined_libdai_lc_h


#include <string>
#include <dai/daialg.h>
#include <dai/enum.h>
#include <dai/factorgraph.h>
#include <dai/properties.h>
#include <dai/exceptions.h>


namespace dai {


/// Approximate inference algorithm "Loop Corrected Belief Propagation" [\ref MoK07]
class LC : public DAIAlgFG {
    private:
        /// Stores for each variable the approximate cavity distribution multiplied with the omitted factors
        std::vector<Factor> _pancakes;
        /// Stores for each variable the approximate cavity distribution
        std::vector<Factor> _cavitydists;
        /// _phis[i][_I] corresponds to \f$ \phi^{\setminus i}_I(x_{I \setminus i}) \f$ in the paper
        std::vector<std::vector<Factor> > _phis;
        /// Single variable beliefs
        std::vector<Factor> _beliefs;
        /// Maximum difference encountered so far
        Real _maxdiff;
        /// Number of iterations needed
        size_t _iters;

    public:
        /// Parameters for LC
        struct Properties {
            /// Enumeration of possible ways to initialize the cavities
            /** The following initialization methods are defined:
             *  - FULL calculates the marginal using calcMarginal()
             *  - PAIR calculates only second order interactions using calcPairBeliefs() with \a accurate == \c false
             *  - PAIR2 calculates only second order interactions using calcPairBeliefs() with \a accurate == \c true
             *  - UNIFORM uses a uniform distribution
             */
            DAI_ENUM(CavityType,FULL,PAIR,PAIR2,UNIFORM);

            /// Enumeration of different update schedules
            /** The following update schedules are defined:
             *  - SEQFIX sequential fixed schedule
             *  - SEQRND sequential random schedule
             */
            DAI_ENUM(UpdateType,SEQFIX,SEQRND);

            /// Verbosity (amount of output sent to stderr)
            size_t verbose;

            /// Maximum number of iterations
            size_t maxiter;

            /// Tolerance for convergence test
            Real tol;

            /// Complete or partial reinitialization of cavity graphs?
            bool reinit;

            /// Damping constant (0.0 means no damping, 1.0 is maximum damping)
            Real damping;

            /// How to initialize the cavities
            CavityType cavity;

            /// What update schedule to use
            UpdateType updates;

            /// Name of the algorithm used to initialize the cavity distributions
            std::string cavainame;

            /// Parameters for the algorithm used to initialize the cavity distributions
            PropertySet cavaiopts;
        } props;

    public:
        /// Default constructor
        LC() : DAIAlgFG(), _pancakes(), _cavitydists(), _phis(), _beliefs(), _maxdiff(), _iters(), props() {}

        /// Construct from FactorGraph \a fg and PropertySet \a opts
        /** \param fg Factor graph.
         *  \param opts Parameters @see Properties
         */
        LC( const FactorGraph &fg, const PropertySet &opts );


    /// \name General InfAlg interface
    //@{
        virtual LC* clone() const { return new LC(*this); }
        virtual LC* construct( const FactorGraph &fg, const PropertySet &opts ) const { return new LC( fg, opts ); }
        virtual std::string name() const { return "LC"; }
        virtual Factor belief( const Var &v ) const { return beliefV( findVar( v ) ); }
        virtual Factor belief( const VarSet &/*vs*/ ) const;
        virtual Factor beliefV( size_t i ) const { return _beliefs[i]; }
        virtual std::vector<Factor> beliefs() const { return _beliefs; }
        virtual Real logZ() const { DAI_THROW(NOT_IMPLEMENTED); return 0.0; }
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

    /// \name Additional interface specific for LC
    //@{
        /// Approximates the cavity distribution of variable \a i, using the inference algorithm \a name with parameters \a opts
        Real CalcCavityDist( size_t i, const std::string &name, const PropertySet &opts );
        /// Approximates all cavity distributions using inference algorithm \a name with parameters \a opts
        Real InitCavityDists( const std::string &name, const PropertySet &opts );
        /// Sets approximate cavity distributions to \a Q
        long SetCavityDists( std::vector<Factor> &Q );
        /// Updates the belief of the Markov blanket of variable \a i based upon the information from its \a _I 'th neighboring factor
        Factor NewPancake (size_t i, size_t _I, bool & hasNaNs);
        /// Calculates the belief of variable \a i
        void CalcBelief (size_t i);
        /// Returns the belief of the Markov blanket of variable \a i (including the variable itself)
        const Factor &pancake (size_t i) const { return _pancakes[i]; };
        /// Returns the approximate cavity distribution for variable \a i
        const Factor &cavitydist (size_t i) const { return _cavitydists[i]; };
    //@}
};


} // end of namespace dai


#endif
