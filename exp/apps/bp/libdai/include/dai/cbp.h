/*  This file is part of libDAI - http://www.libdai.org/
 *
 *  Copyright (c) 2006-2011, The libDAI authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
 */


/// \file
/// \brief Defines class CBP, which implements Conditioned Belief Propagation


#ifndef __defined_libdai_cbp_h
#define __defined_libdai_cbp_h


#include <fstream>
#include <boost/shared_ptr.hpp>

#include <dai/daialg.h>
#include <dai/bbp.h>


namespace dai {


/// Class for CBP (Conditioned Belief Propagation) [\ref EaG09]
/** This approximate inference algorithm uses configurable heuristics to choose a variable
 *  \f$ x_i \f$ and a state \f$ x_i^* \f$. Inference is done with \f$ x_i \f$ "clamped" to \f$ x_i^* \f$
 *  (i.e., conditional on \f$ x_i = x_i^* \f$), and also with the negation of this condition. 
 *  Clamping is done recursively up to a fixed number of levels (other stopping criteria are 
 *  also implemented, see the CBP::Properties::RecurseType property). The resulting approximate 
 *  marginals are combined using estimates of the partition sum.
 *
 *  \author Frederik Eaton
 */
class CBP : public DAIAlgFG {
    private:
        /// Variable beliefs
        std::vector<Factor> _beliefsV;
        /// Factor beliefs
        std::vector<Factor> _beliefsF;
        /// Logarithm of partition sum
        Real _logZ;

        /// Numer of iterations needed
        size_t _iters;
        /// Maximum difference encountered so far
        Real _maxdiff;

        /// Number of clampings at each leaf node
        Real _sum_level;
        /// Number of leaves of recursion tree
        size_t _num_leaves;

        /// Output stream where information about the clampings is written
        boost::shared_ptr<std::ofstream> _clamp_ofstream;


    public:
        /// Default constructor
        CBP() : DAIAlgFG(), _beliefsV(), _beliefsF(), _logZ(0.0), _iters(0), _maxdiff(0.0), _sum_level(0.0), _num_leaves(0), _clamp_ofstream() {}

        /// Construct CBP object from FactorGraph \a fg and PropertySet \a opts
        /** \param fg Factor graph.
         *  \param opts Parameters @see Properties
         */
        CBP( const FactorGraph &fg, const PropertySet &opts ) : DAIAlgFG(fg) {
            props.set( opts );
            construct();
        }

    /// \name General InfAlg interface
    //@{
        virtual CBP* clone() const { return new CBP(*this); }
        virtual CBP* construct( const FactorGraph &fg, const PropertySet &opts ) const { return new CBP( fg, opts ); }
        virtual std::string name() const { return "CBP"; }
        virtual Factor belief( const Var &v ) const { return beliefV( findVar( v ) ); }
        virtual Factor belief( const VarSet & ) const { DAI_THROW(NOT_IMPLEMENTED); }
        virtual Factor beliefV( size_t i ) const { return _beliefsV[i]; }
        virtual Factor beliefF( size_t I ) const { return _beliefsF[I]; }
        virtual std::vector<Factor> beliefs() const { return concat(_beliefsV, _beliefsF); }
        virtual Real logZ() const { return _logZ; }
        virtual void init() {};
        virtual void init( const VarSet & ) {};
        virtual Real run();
        virtual Real maxDiff() const { return _maxdiff; }
        virtual size_t Iterations() const { return _iters; }
        virtual void setMaxIter( size_t maxiter ) { props.maxiter = maxiter; }
        virtual void setProperties( const PropertySet &opts ) { props.set( opts ); }
        virtual PropertySet getProperties() const { return props.get(); }
        virtual std::string printProperties() const { return props.toString(); }
    //@}

        //----------------------------------------------------------------

        /// Parameters for CBP
        /* PROPERTIES(props,CBP) {
            /// Enumeration of possible update schedules
            typedef BP::Properties::UpdateType UpdateType;
            /// Enumeration of possible methods for deciding when to stop recursing
            DAI_ENUM(RecurseType,REC_FIXED,REC_LOGZ,REC_BDIFF);
            /// Enumeration of possible heuristics for choosing clamping variable
            DAI_ENUM(ChooseMethodType,CHOOSE_RANDOM,CHOOSE_MAXENT,CHOOSE_BBP,CHOOSE_BP_L1,CHOOSE_BP_CFN);
            /// Enumeration of possible clampings: variables or factors
            DAI_ENUM(ClampType,CLAMP_VAR,CLAMP_FACTOR);

            /// Verbosity (amount of output sent to stderr)
            size_t verbose = 0;

            /// Tolerance for BP convergence test
            Real tol;
            /// Update style for BP
            UpdateType updates;
            /// Maximum number of iterations for BP
            size_t maxiter;

            /// Tolerance used for controlling recursion depth (\a recurse is REC_LOGZ or REC_BDIFF)
            Real rec_tol;
            /// Maximum number of levels of recursion (\a recurse is REC_FIXED)
            size_t max_levels = 10;
            /// If choose==CHOOSE_BBP and maximum adjoint is less than this value, don't recurse
            Real min_max_adj;
            /// Heuristic for choosing clamping variable
            ChooseMethodType choose;
            /// Method for deciding when to stop recursing
            RecurseType recursion;
            /// Whether to clamp variables or factors
            ClampType clamp;
            /// Properties to pass to BBP
            PropertySet bbp_props;
            /// Cost function to use for BBP
            BBPCostFunction bbp_cfn;
            /// Random seed
            size_t rand_seed = 0;

            /// If non-empty, write clamping choices to this file
            std::string clamp_outfile = "";
        }
        */
/* {{{ GENERATED CODE: DO NOT EDIT. Created by
    ./scripts/regenerate-properties include/dai/cbp.h src/cbp.cpp
*/
        struct Properties {
            /// Enumeration of possible update schedules
            typedef BP::Properties::UpdateType UpdateType;
            /// Enumeration of possible methods for deciding when to stop recursing
            DAI_ENUM(RecurseType,REC_FIXED,REC_LOGZ,REC_BDIFF);
            /// Enumeration of possible heuristics for choosing clamping variable
            DAI_ENUM(ChooseMethodType,CHOOSE_RANDOM,CHOOSE_MAXENT,CHOOSE_BBP,CHOOSE_BP_L1,CHOOSE_BP_CFN);
            /// Enumeration of possible clampings: variables or factors
            DAI_ENUM(ClampType,CLAMP_VAR,CLAMP_FACTOR);
            /// Verbosity (amount of output sent to stderr)
            size_t verbose;
            /// Tolerance for BP convergence test
            Real tol;
            /// Update style for BP
            UpdateType updates;
            /// Maximum number of iterations for BP
            size_t maxiter;
            /// Tolerance used for controlling recursion depth (\a recurse is REC_LOGZ or REC_BDIFF)
            Real rec_tol;
            /// Maximum number of levels of recursion (\a recurse is REC_FIXED)
            size_t max_levels;
            /// If choose==CHOOSE_BBP and maximum adjoint is less than this value, don't recurse
            Real min_max_adj;
            /// Heuristic for choosing clamping variable
            ChooseMethodType choose;
            /// Method for deciding when to stop recursing
            RecurseType recursion;
            /// Whether to clamp variables or factors
            ClampType clamp;
            /// Properties to pass to BBP
            PropertySet bbp_props;
            /// Cost function to use for BBP
            BBPCostFunction bbp_cfn;
            /// Random seed
            size_t rand_seed;
            /// If non-empty, write clamping choices to this file
            std::string clamp_outfile;

            /// Set members from PropertySet
            /** \throw UNKNOWN_PROPERTY if a Property key is not recognized
             *  \throw NOT_ALL_PROPERTIES_SPECIFIED if an expected Property is missing
             */
            void set(const PropertySet &opts);
            /// Get members into PropertySet
            PropertySet get() const;
            /// Convert to a string which can be parsed as a PropertySet
            std::string toString() const;
        } props;
/* }}} END OF GENERATED CODE */

    private:
        /// Prints beliefs, variables and partition sum, in case of a debugging build
        void printDebugInfo();

        /// Called by run(), and by itself. Implements the main algorithm.
        /** Chooses a variable to clamp, recurses, combines the partition sum 
         *  and belief estimates of the children, and returns the improved
         *  estimates in \a lz_out and \a beliefs_out to its parent.
         */
        void runRecurse( InfAlg *bp, Real orig_logZ, std::vector<size_t> clamped_vars_list, size_t &num_leaves,
                         size_t &choose_count, Real &sum_level, Real &lz_out, std::vector<Factor> &beliefs_out );

        /// Choose the next variable to clamp.
        /** Choose the next variable to clamp, given a converged InfAlg \a bp,
         *  and a vector of variables that are already clamped (\a
         *  clamped_vars_list). Returns the chosen variable in \a i, and
         *  the set of states in \a xis. If \a maxVarOut is non-NULL and
         *  \a props.choose == \c CHOOSE_BBP then it is used to store the
         *  adjoint of the chosen variable.
         */
        virtual bool chooseNextClampVar( InfAlg* bp, std::vector<size_t> &clamped_vars_list, size_t &i, std::vector<size_t> &xis, Real *maxVarOut );

        /// Return the InfAlg to use at each step of the recursion.
        /** \todo At present, CBP::getInfAlg() only returns a BP instance; 
         *  it should be possible to select other inference algorithms via a property
         */
        InfAlg* getInfAlg();

        /// Sets variable beliefs, factor beliefs and log partition sum to the specified values
        /** \param bs should be a concatenation of the variable beliefs followed by the factor beliefs
         *  \param logZ log partition sum
         */
        void setBeliefs( const std::vector<Factor> &bs, Real logZ );

        /// Constructor helper function
        void construct();
};


/// Find the best variable/factor to clamp using BBP.
/** Takes a converged inference algorithm as input, runs Gibbs and BP_dual, creates
 *  and runs a BBP object, finds the best variable/factor (the one with the maximum
 *  factor adjoint), and returns the corresponding (index,state) pair.
 *  \param in_bp inference algorithm (compatible with BP) that should have converged;
 *  \param clampingVar if \c true, finds best variable, otherwise, finds best factor;
 *  \param bbp_props BBP parameters to use;
 *  \param cfn BBP cost function to use;
 *  \param maxVarOut maximum adjoint value (only set if not NULL).
 *  \see BBP
 *  \relates CBP
 */
std::pair<size_t, size_t> BBPFindClampVar( const InfAlg &in_bp, bool clampingVar, const PropertySet &bbp_props, const BBPCostFunction &cfn, Real *maxVarOut );


} // end of namespace dai


#endif
