/*  This file is part of libDAI - http://www.libdai.org/
 *
 *  Copyright (c) 2006-2011, The libDAI authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
 */


/// \file
/// \brief Defines class BBP, which implements Back-Belief-Propagation
/// \todo Clean up code


#ifndef ___defined_libdai_bbp_h
#define ___defined_libdai_bbp_h


#include <vector>
#include <utility>

#include <dai/prob.h>
#include <dai/daialg.h>
#include <dai/factorgraph.h>
#include <dai/enum.h>
#include <dai/bp_dual.h>


namespace dai {


/// Enumeration of several cost functions that can be used with BBP
/** \note This class is meant as a base class for BBPCostFunction, which provides additional functionality.
 */
DAI_ENUM(BBPCostFunctionBase,CFN_GIBBS_B,CFN_GIBBS_B2,CFN_GIBBS_EXP,CFN_GIBBS_B_FACTOR,CFN_GIBBS_B2_FACTOR,CFN_GIBBS_EXP_FACTOR,CFN_VAR_ENT,CFN_FACTOR_ENT,CFN_BETHE_ENT);


/// Predefined cost functions that can be used with BBP
class BBPCostFunction : public BBPCostFunctionBase {
    public:
        /// Default constructor
        BBPCostFunction() : BBPCostFunctionBase() {}

        /// Construct from BBPCostFunctionBase \a x
        BBPCostFunction( const BBPCostFunctionBase &x ) : BBPCostFunctionBase(x) {}

        /// Returns whether this cost function depends on having a Gibbs state
        bool needGibbsState() const;

        /// Evaluates cost function in state \a stateP using the information in inference algorithm \a ia
        Real evaluate( const InfAlg &ia, const std::vector<size_t> *stateP ) const;

        /// Assignment operator
        BBPCostFunction& operator=( const BBPCostFunctionBase &x ) {
            BBPCostFunctionBase::operator=( x );
            return *this;
        }
};


/// Implements BBP (Back-Belief-Propagation) [\ref EaG09]
/** \author Frederik Eaton
 */
class BBP {
    private:
    /// \name Input variables
    //@{
        /// Stores a BP_dual helper object
        BP_dual _bp_dual;
        /// Pointer to the factor graph
        const FactorGraph *_fg;
        /// Pointer to the approximate inference algorithm (currently, only BP objects are supported)
        const InfAlg *_ia;
    //@}

    /// \name Output variables
    //@{
        /// Variable factor adjoints
        std::vector<Prob> _adj_psi_V;
        /// Factor adjoints
        std::vector<Prob> _adj_psi_F;
        /// Variable->factor message adjoints (indexed [i][_I])
        std::vector<std::vector<Prob> > _adj_n;
        /// Factor->variable message adjoints (indexed [i][_I])
        std::vector<std::vector<Prob> > _adj_m;
        /// Normalized variable belief adjoints
        std::vector<Prob> _adj_b_V;
        /// Normalized factor belief adjoints
        std::vector<Prob> _adj_b_F;
    //@}

    /// \name Internal state variables
    //@{
        /// Initial variable factor adjoints
        std::vector<Prob> _init_adj_psi_V;
        /// Initial factor adjoints
        std::vector<Prob> _init_adj_psi_F;

        /// Unnormalized variable->factor message adjoint (indexed [i][_I])
        std::vector<std::vector<Prob> > _adj_n_unnorm;
        /// Unnormalized factor->variable message adjoint (indexed [i][_I])
        std::vector<std::vector<Prob> > _adj_m_unnorm;
        /// Updated normalized variable->factor message adjoint (indexed [i][_I])
        std::vector<std::vector<Prob> > _new_adj_n;
        /// Updated normalized factor->variable message adjoint (indexed [i][_I])
        std::vector<std::vector<Prob> > _new_adj_m;
        /// Unnormalized variable belief adjoints
        std::vector<Prob> _adj_b_V_unnorm;
        /// Unnormalized factor belief adjoints
        std::vector<Prob> _adj_b_F_unnorm;

        /// _Tmsg[i][_I] (see eqn. (41) in [\ref EaG09])
        std::vector<std::vector<Prob > > _Tmsg;
        /// _Umsg[I][_i] (see eqn. (42) in [\ref EaG09])
        std::vector<std::vector<Prob > > _Umsg;
        /// _Smsg[i][_I][_j] (see eqn. (43) in [\ref EaG09])
        std::vector<std::vector<std::vector<Prob > > > _Smsg;
        /// _Rmsg[I][_i][_J] (see eqn. (44) in [\ref EaG09])
        std::vector<std::vector<std::vector<Prob > > > _Rmsg;

        /// Number of iterations done
        size_t _iters;
    //@}

    /// \name Index cache management (for performance)
    //@{
        /// Index type
        typedef std::vector<size_t>  _ind_t;
        /// Cached indices (indexed [i][_I])
        std::vector<std::vector<_ind_t> >  _indices;
        /// Prepares index cache _indices
        /** \see bp.cpp
         */
        void RegenerateInds();
        /// Returns an index from the cache
        const _ind_t& _index(size_t i, size_t _I) const { return _indices[i][_I]; }
    //@}

    /// \name Initialization helper functions
    //@{
        /// Calculate T values; see eqn. (41) in [\ref EaG09]
        void RegenerateT();
        /// Calculate U values; see eqn. (42) in [\ref EaG09]
        void RegenerateU();
        /// Calculate S values; see eqn. (43) in [\ref EaG09]
        void RegenerateS();
        /// Calculate R values; see eqn. (44) in [\ref EaG09]
        void RegenerateR();
        /// Calculate _adj_b_V_unnorm and _adj_b_F_unnorm from _adj_b_V and _adj_b_F
        void RegenerateInputs();
        /// Initialise members for factor adjoints
        /** \pre RegenerateInputs() should be called first
         */
        void RegeneratePsiAdjoints();
        /// Initialise members for message adjoints for parallel algorithm
        /** \pre RegenerateInputs() should be called first
         */
        void RegenerateParMessageAdjoints();
        /// Initialise members for message adjoints for sequential algorithm
        /** Same as RegenerateMessageAdjoints, but calls sendSeqMsgN rather
         *  than updating _adj_n (and friends) which are unused in the sequential algorithm.
         *  \pre RegenerateInputs() should be called first
         */
        void RegenerateSeqMessageAdjoints();
        /// Called by \a init, recalculates intermediate values
        void Regenerate();
    //@}

    /// \name Accessors/mutators
    //@{
        /// Returns reference to T value; see eqn. (41) in [\ref EaG09]
        Prob & T(size_t i, size_t _I) { return _Tmsg[i][_I]; }
        /// Returns constant reference to T value; see eqn. (41) in [\ref EaG09]
        const Prob & T(size_t i, size_t _I) const { return _Tmsg[i][_I]; }
        /// Returns reference to U value; see eqn. (42) in [\ref EaG09]
        Prob & U(size_t I, size_t _i) { return _Umsg[I][_i]; }
        /// Returns constant reference to U value; see eqn. (42) in [\ref EaG09]
        const Prob & U(size_t I, size_t _i) const { return _Umsg[I][_i]; }
        /// Returns reference to S value; see eqn. (43) in [\ref EaG09]
        Prob & S(size_t i, size_t _I, size_t _j) { return _Smsg[i][_I][_j]; }
        /// Returns constant reference to S value; see eqn. (43) in [\ref EaG09]
        const Prob & S(size_t i, size_t _I, size_t _j) const { return _Smsg[i][_I][_j]; }
        /// Returns reference to R value; see eqn. (44) in [\ref EaG09]
        Prob & R(size_t I, size_t _i, size_t _J) { return _Rmsg[I][_i][_J]; }
        /// Returns constant reference to R value; see eqn. (44) in [\ref EaG09]
        const Prob & R(size_t I, size_t _i, size_t _J) const { return _Rmsg[I][_i][_J]; }

        /// Returns reference to variable->factor message adjoint
        Prob& adj_n(size_t i, size_t _I) { return _adj_n[i][_I]; }
        /// Returns constant reference to variable->factor message adjoint
        const Prob& adj_n(size_t i, size_t _I) const { return _adj_n[i][_I]; }
        /// Returns reference to factor->variable message adjoint
        Prob& adj_m(size_t i, size_t _I) { return _adj_m[i][_I]; }
        /// Returns constant reference to factor->variable message adjoint
        const Prob& adj_m(size_t i, size_t _I) const { return _adj_m[i][_I]; }
    //@}

    /// \name Parallel algorithm
    //@{
        /// Calculates new variable->factor message adjoint
        /** Increases variable factor adjoint according to eqn. (27) in [\ref EaG09] and
         *  calculates the new variable->factor message adjoint according to eqn. (29) in [\ref EaG09].
         */
        void calcNewN( size_t i, size_t _I );
        /// Calculates new factor->variable message adjoint
        /** Increases factor adjoint according to eqn. (28) in [\ref EaG09] and
         *  calculates the new factor->variable message adjoint according to the r.h.s. of eqn. (30) in [\ref EaG09].
         */
        void calcNewM( size_t i, size_t _I );
        /// Calculates unnormalized variable->factor message adjoint from the normalized one
        void calcUnnormMsgN( size_t i, size_t _I );
        /// Calculates unnormalized factor->variable message adjoint from the normalized one
        void calcUnnormMsgM( size_t i, size_t _I );
        /// Updates (un)normalized variable->factor message adjoints
        void upMsgN( size_t i, size_t _I );
        /// Updates (un)normalized factor->variable message adjoints
        void upMsgM( size_t i, size_t _I );
        /// Do one parallel update of all message adjoints
        void doParUpdate();
    //@}

    /// \name Sequential algorithm
    //@{
        /// Helper function for sendSeqMsgM(): increases factor->variable message adjoint by \a p and calculates the corresponding unnormalized adjoint
        void incrSeqMsgM( size_t i, size_t _I, const Prob& p );
        //  DISABLED BECAUSE IT IS BUGGY:
        //  void updateSeqMsgM( size_t i, size_t _I );
        /// Sets normalized factor->variable message adjoint and calculates the corresponding unnormalized adjoint
        void setSeqMsgM( size_t i, size_t _I, const Prob &p );
        /// Implements routine Send-n in Figure 5 in [\ref EaG09]
        void sendSeqMsgN( size_t i, size_t _I, const Prob &f );
        /// Implements routine Send-m in Figure 5 in [\ref EaG09]
        void sendSeqMsgM( size_t i, size_t _I );
    //@}

        /// Computes the adjoint of the unnormed probability vector from the normalizer and the adjoint of the normalized probability vector
        /** \see eqn. (13) in [\ref EaG09]
         */
        Prob unnormAdjoint( const Prob &w, Real Z_w, const Prob &adj_w );

        /// Calculates averaged L1 norm of unnormalized message adjoints
        Real getUnMsgMag();
        /// Calculates averaged L1 norms of current and new normalized message adjoints
        void getMsgMags( Real &s, Real &new_s );
        /// Returns indices and magnitude of the largest normalized factor->variable message adjoint
        void getArgmaxMsgM( size_t &i, size_t &_I, Real &mag );
        /// Returns magnitude of the largest (in L1-norm) normalized factor->variable message adjoint
        Real getMaxMsgM();

        /// Calculates sum of L1 norms of all normalized factor->variable message adjoints
        Real getTotalMsgM();
        /// Calculates sum of L1 norms of all updated normalized factor->variable message adjoints
        Real getTotalNewMsgM();
        /// Calculates sum of L1 norms of all normalized variable->factor message adjoints
        Real getTotalMsgN();

        /// Returns a vector of Probs (filled with zeroes) with state spaces corresponding to the factors in the factor graph \a fg
        std::vector<Prob> getZeroAdjF( const FactorGraph &fg );
        /// Returns a vector of Probs (filled with zeroes) with state spaces corresponding to the variables in the factor graph \a fg
        std::vector<Prob> getZeroAdjV( const FactorGraph &fg );

    public:
    /// \name Constructors/destructors
    //@{
        /// Construct BBP object from a InfAlg \a ia and a PropertySet \a opts
        /** \param ia should be a BP object or something compatible
         *  \param opts Parameters @see Properties
         */
        BBP( const InfAlg *ia, const PropertySet &opts ) : _bp_dual(ia), _fg(&(ia->fg())), _ia(ia) {
            props.set(opts);
        }
    //@}

    /// \name Initialization
    //@{
        /// Initializes from given belief adjoints \a adj_b_V, \a adj_b_F and initial factor adjoints \a adj_psi_V, \a adj_psi_F
        void init( const std::vector<Prob> &adj_b_V, const std::vector<Prob> &adj_b_F, const std::vector<Prob> &adj_psi_V, const std::vector<Prob> &adj_psi_F ) {
            _adj_b_V = adj_b_V;
            _adj_b_F = adj_b_F;
            _init_adj_psi_V = adj_psi_V;
            _init_adj_psi_F = adj_psi_F;
            Regenerate();
        }

        /// Initializes from given belief adjoints \a adj_b_V and \a adj_b_F (setting initial factor adjoints to zero)
        void init( const std::vector<Prob> &adj_b_V, const std::vector<Prob> &adj_b_F ) {
            init( adj_b_V, adj_b_F, getZeroAdjV(*_fg), getZeroAdjF(*_fg) );
        }

        /// Initializes variable belief adjoints \a adj_b_V (and sets factor belief adjoints and initial factor adjoints to zero)
        void init_V( const std::vector<Prob> &adj_b_V ) {
            init( adj_b_V, getZeroAdjF(*_fg) );
        }

        /// Initializes factor belief adjoints \a adj_b_F (and sets variable belief adjoints and initial factor adjoints to zero)
        void init_F( const std::vector<Prob> &adj_b_F ) {
            init( getZeroAdjV(*_fg), adj_b_F );
        }

        /// Initializes with adjoints calculated from cost function \a cfn, and state \a stateP
        /** Uses the internal pointer to an inference algorithm in combination with the cost function and state for initialization.
         *  \param cfn Cost function used for initialization;
         *  \param stateP is a Gibbs state and can be NULL; it will be initialised using a Gibbs run.
         */
        void initCostFnAdj( const BBPCostFunction &cfn, const std::vector<size_t> *stateP );
    //@}

    /// \name BBP Algorithm
    //@{
        /// Perform iterative updates until change is less than given tolerance
        void run();
    //@}

    /// \name Query results
    //@{
        /// Returns reference to variable factor adjoint
        Prob& adj_psi_V(size_t i) { return _adj_psi_V[i]; }
        /// Returns constant reference to variable factor adjoint
        const Prob& adj_psi_V(size_t i) const { return _adj_psi_V[i]; }
        /// Returns reference to factor adjoint
        Prob& adj_psi_F(size_t I) { return _adj_psi_F[I]; }
        /// Returns constant reference to factor adjoint
        const Prob& adj_psi_F(size_t I) const { return _adj_psi_F[I]; }
        /// Returns reference to variable belief adjoint
        Prob& adj_b_V(size_t i) { return _adj_b_V[i]; }
        /// Returns constant reference to variable belief adjoint
        const Prob& adj_b_V(size_t i) const { return _adj_b_V[i]; }
        /// Returns reference to factor belief adjoint
        Prob& adj_b_F(size_t I) { return _adj_b_F[I]; }
        /// Returns constant reference to factor belief adjoint
        const Prob& adj_b_F(size_t I) const { return _adj_b_F[I]; }
        /// Return number of iterations done so far
        size_t Iterations() { return _iters; }
    //@}

    public:
        /// Parameters for BBP
        /* PROPERTIES(props,BBP) {
           /// Enumeration of possible update schedules
           /// The following update schedules are defined:
           /// - SEQ_FIX fixed sequential updates
           /// - SEQ_MAX maximum residual updates (inspired by [\ref EMK06])
           /// - SEQ_BP_REV schedule used by BP, but reversed
           /// - SEQ_BP_FWD schedule used by BP
           /// - PAR parallel updates
           DAI_ENUM(UpdateType,SEQ_FIX,SEQ_MAX,SEQ_BP_REV,SEQ_BP_FWD,PAR);

           /// Verbosity (amount of output sent to stderr)
           size_t verbose = 0;

           /// Maximum number of iterations
           size_t maxiter;

           /// Tolerance for convergence test
           /// \note Not used for updates = SEQ_BP_REV, SEQ_BP_FWD
           Real tol;

           /// Damping constant (0 for none); damping = 1 - lambda where lambda is the damping constant used in [\ref EaG09]
           Real damping;

           /// Update schedule
           UpdateType updates;

           // DISABLED BECAUSE IT IS BUGGY:
           // bool clean_updates;
        }
        */
/* {{{ GENERATED CODE: DO NOT EDIT. Created by
    ./scripts/regenerate-properties include/dai/bbp.h src/bbp.cpp
*/
        struct Properties {
            /// Enumeration of possible update schedules
            /** The following update schedules are defined:
             *  - SEQ_FIX fixed sequential updates
             *  - SEQ_MAX maximum residual updates (inspired by [\ref EMK06])
             *  - SEQ_BP_REV schedule used by BP, but reversed
             *  - SEQ_BP_FWD schedule used by BP
             *  - PAR parallel updates
             */
            DAI_ENUM(UpdateType,SEQ_FIX,SEQ_MAX,SEQ_BP_REV,SEQ_BP_FWD,PAR);
            /// Verbosity (amount of output sent to stderr)
            size_t verbose;
            /// Maximum number of iterations
            size_t maxiter;
            /// Tolerance for convergence test
            /** \note Not used for updates = SEQ_BP_REV, SEQ_BP_FWD
             */
            Real tol;
            /// Damping constant (0 for none); damping = 1 - lambda where lambda is the damping constant used in [\ref EaG09]
            Real damping;
            /// Update schedule
            UpdateType updates;

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
};


/// Function to verify the validity of adjoints computed by BBP using numerical differentiation.
/** Factors containing a variable are multiplied by small adjustments to verify accuracy of calculated variable factor adjoints.
 *  \param bp BP object;
 *  \param state Global state of all variables;
 *  \param bbp_props BBP parameters;
 *  \param cfn Cost function to be used;
 *  \param h Size of perturbation.
 *  \relates BBP
 */
Real numericBBPTest( const InfAlg &bp, const std::vector<size_t> *state, const PropertySet &bbp_props, const BBPCostFunction &cfn, Real h );


} // end of namespace dai


#endif
