/*  This file is part of libDAI - http://www.libdai.org/
 *
 *  Copyright (c) 2006-2011, The libDAI authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
 */


#ifndef __defined_libdai_emalg_h
#define __defined_libdai_emalg_h


#include <vector>
#include <map>

#include <dai/factor.h>
#include <dai/daialg.h>
#include <dai/evidence.h>
#include <dai/index.h>
#include <dai/properties.h>


/// \file
/// \brief Defines classes related to Expectation Maximization (EMAlg, ParameterEstimation, CondProbEstimation and SharedParameters)
/// \todo Implement parameter estimation for undirected models / factor graphs.


namespace dai {


/// Base class for parameter estimation methods.
/** This class defines the general interface of parameter estimation methods.
 *
 *  Implementations of this interface (see e.g. CondProbEstimation) should 
 *  register a factory function (virtual constructor) via the static 
 *  registerMethod() function.
 *  This factory function should return a pointer to a newly constructed 
 *  object, whose type is a subclass of ParameterEstimation, and gets as 
 *  input a PropertySet of parameters. After a subclass has been registered, 
 *  instances of it can be constructed using the construct() method.
 *
 *  Implementations are responsible for collecting data from a probability
 *  vector passed to it from a SharedParameters container object.
 *
 *  The default registry only contains CondProbEstimation, named
 *  "CondProbEstimation".
 *
 *  \author Charles Vaske
 */
class ParameterEstimation {
    public:
        /// Type of pointer to factory function.
        typedef ParameterEstimation* (*ParamEstFactory)( const PropertySet& );

        /// Virtual destructor for deleting pointers to derived classes.
        virtual ~ParameterEstimation() {}

        /// Virtual copy constructor.
        virtual ParameterEstimation* clone() const = 0;

        /// General factory method that constructs the desired ParameterEstimation subclass
        /** \param method Name of the subclass that should be constructed;
         *  \param p Parameters passed to constructor of subclass.
         *  \note \a method should either be in the default registry or should be registered first using registerMethod().
         *  \throw UNKNOWN_PARAMETER_ESTIMATION_METHOD if the requested \a method is not registered.
         */
        static ParameterEstimation* construct( const std::string &method, const PropertySet &p );

        /// Register a subclass so that it can be used with construct().
        static void registerMethod( const std::string &method, const ParamEstFactory &f ) {
            if( _registry == NULL )
                loadDefaultRegistry();
            (*_registry)[method] = f;
        }

        /// Estimate the factor using the accumulated sufficient statistics and reset.
        virtual Prob estimate() = 0;

        /// Accumulate the sufficient statistics for \a p.
        virtual void addSufficientStatistics( const Prob &p ) = 0;

        /// Returns the size of the Prob that should be passed to addSufficientStatistics.
        virtual size_t probSize() const = 0;

    private:
        /// A static registry containing all methods registered so far.
        static std::map<std::string, ParamEstFactory> *_registry;

        /// Registers default ParameterEstimation subclasses (currently, only CondProbEstimation).
        static void loadDefaultRegistry();
};


/// Estimates the parameters of a conditional probability table, using pseudocounts.
/** \author Charles Vaske
 */
class CondProbEstimation : private ParameterEstimation {
    private:
        /// Number of states of the variable of interest
        size_t _target_dim;
        /// Current pseudocounts
        Prob _stats;
        /// Initial pseudocounts
        Prob _initial_stats;

    public:
        /// Constructor
        /** For a conditional probability \f$ P( X | Y ) \f$,
         *  \param target_dimension should equal \f$ | X | \f$
         *  \param pseudocounts are the initial pseudocounts, of length \f$ |X| \cdot |Y| \f$
         */
        CondProbEstimation( size_t target_dimension, const Prob &pseudocounts );

        /// Virtual constructor, using a PropertySet.
        /** Some keys in the PropertySet are required.
         *  For a conditional probability \f$ P( X | Y ) \f$,
         *     - \a target_dimension should be equal to \f$ | X | \f$
         *     - \a total_dimension should be equal to \f$ |X| \cdot |Y| \f$
         *
         *  An optional key is:
         *     - \a pseudo_count which specifies the initial counts (defaults to 1)
         */
        static ParameterEstimation* factory( const PropertySet &p );

        /// Virtual copy constructor
        virtual ParameterEstimation* clone() const { return new CondProbEstimation( _target_dim, _initial_stats ); }

        /// Virtual destructor
        virtual ~CondProbEstimation() {}

        /// Returns an estimate of the conditional probability distribution.
        /** The format of the resulting Prob keeps all the values for
         *  \f$ P(X | Y=y) \f$ in sequential order in the array.
         */
        virtual Prob estimate();

        /// Accumulate sufficient statistics from the expectations in \a p
        virtual void addSufficientStatistics( const Prob &p );

        /// Returns the required size for arguments to addSufficientStatistics().
        virtual size_t probSize() const { return _stats.size(); }
};


/// Represents a single factor or set of factors whose parameters should be estimated.
/** To ensure that parameters can be shared between different factors during
 *  EM learning, each factor's values are reordered to match a desired variable
 *  ordering. The ordering of the variables in a factor may therefore differ
 *  from the canonical ordering used in libDAI. The SharedParameters
 *  class combines one or more factors (together with the specified orderings
 *  of the variables) with a ParameterEstimation object, taking care of the
 *  necessary permutations of the factor entries / parameters.
 * 
 *  \author Charles Vaske
 */
class SharedParameters {
    public:
        /// Convenience label for an index of a factor in a FactorGraph.
        typedef size_t FactorIndex;
        /// Convenience label for a grouping of factor orientations.
        typedef std::map<FactorIndex, std::vector<Var> > FactorOrientations;

    private:
        /// Maps factor indices to the corresponding VarSets
        std::map<FactorIndex, VarSet> _varsets;
        /// Maps factor indices to the corresponding Permute objects that permute the canonical ordering into the desired ordering
        std::map<FactorIndex, Permute> _perms;
        /// Maps factor indices to the corresponding desired variable orderings
        FactorOrientations _varorders;
        /// Parameter estimation method to be used
        ParameterEstimation *_estimation;
        /// Indicates whether \c *this gets ownership of _estimation
        bool _ownEstimation;

        /// Calculates the permutation that permutes the canonical ordering into the desired ordering
        /** \param varOrder Desired ordering of variables
         *  \param outVS Contains variables in \a varOrder represented as a VarSet
         *  \return Permute object for permuting variables in varOrder from the canonical libDAI ordering into the desired ordering
         */
        static Permute calculatePermutation( const std::vector<Var> &varOrder, VarSet &outVS );

        /// Initializes _varsets and _perms from _varorders and checks whether their state spaces correspond with _estimation.probSize()
        void setPermsAndVarSetsFromVarOrders();

    public:
        /// Constructor
        /** \param varorders  all the factor orientations for this parameter
         *  \param estimation a pointer to the parameter estimation method
         *  \param ownPE whether the constructed object gets ownership of \a estimation
         */
        SharedParameters( const FactorOrientations &varorders, ParameterEstimation *estimation, bool ownPE=false );

        /// Construct a SharedParameters object from an input stream \a is and a factor graph \a fg
        /** \see \ref fileformats-emalg-sharedparameters
         *  \throw INVALID_EMALG_FILE if the input stream is not valid
         */
        SharedParameters( std::istream &is, const FactorGraph &fg );

        /// Copy constructor
        SharedParameters( const SharedParameters &sp ) : _varsets(sp._varsets), _perms(sp._perms), _varorders(sp._varorders), _estimation(sp._estimation), _ownEstimation(sp._ownEstimation) {
            // If sp owns its _estimation object, we should clone it instead of copying the pointer
            if( _ownEstimation )
                _estimation = _estimation->clone();
        }

        /// Destructor
        ~SharedParameters() {
            // If we own the _estimation object, we should delete it now
            if( _ownEstimation )
                delete _estimation;
        }

        /// Collect the sufficient statistics from expected values (beliefs) according to \a alg
        /** For each of the relevant factors (that shares the parameters we are interested in),
         *  the corresponding belief according to \a alg is obtained and its entries are permuted
         *  such that their ordering corresponds with the shared parameters that we are estimating.
         *  Then, the parameter estimation subclass method addSufficientStatistics() is called with
         *  this vector of expected values of the parameters as input.
         */
        void collectSufficientStatistics( InfAlg &alg );

        /// Estimate and set the shared parameters
        /** Based on the sufficient statistics collected so far, the shared parameters are estimated
         *  using the parameter estimation subclass method estimate(). Then, each of the relevant
         *  factors in \a fg (that shares the parameters we are interested in) is set according 
         *  to those parameters (permuting the parameters accordingly).
         */
        void setParameters( FactorGraph &fg );
};


/// A MaximizationStep groups together several parameter estimation tasks (SharedParameters objects) into a single unit.
/** \author Charles Vaske
 */
class MaximizationStep {
    private:
        /// Vector of parameter estimation tasks of which this maximization step consists
        std::vector<SharedParameters> _params;

    public:
        /// Default constructor
        MaximizationStep() : _params() {}

        /// Construct MaximizationStep from a vector of parameter estimation tasks
        MaximizationStep( std::vector<SharedParameters> &maximizations ) : _params(maximizations) {}

        /// Constructor from an input stream and a corresponding factor graph
        /** \see \ref fileformats-emalg-maximizationstep
         */
        MaximizationStep( std::istream &is, const FactorGraph &fg_varlookup );

        /// Collect the beliefs from this InfAlg as expectations for the next Maximization step
        void addExpectations( InfAlg &alg );

        /// Using all of the currently added expectations, make new factors with maximized parameters and set them in the FactorGraph.
        void maximize( FactorGraph &fg );

    /// \name Iterator interface
    //@{
        /// Iterator over the parameter estimation tasks
        typedef std::vector<SharedParameters>::iterator iterator;
        /// Constant iterator over the parameter estimation tasks
        typedef std::vector<SharedParameters>::const_iterator const_iterator;

        /// Returns iterator that points to the first parameter estimation task
        iterator begin() { return _params.begin(); }
        /// Returns constant iterator that points to the first parameter estimation task
        const_iterator begin() const { return _params.begin(); }
        /// Returns iterator that points beyond the last parameter estimation task
        iterator end() { return _params.end(); }
        /// Returns constant iterator that points beyond the last parameter estimation task
        const_iterator end() const { return _params.end(); }
    //@}
};


/// EMAlg performs Expectation Maximization to learn factor parameters.
/** This requires specifying:
 *     - Evidence (instances of observations from the graphical model),
 *     - InfAlg for performing the E-step (which includes the factor graph),
 *     - a vector of MaximizationStep 's steps to be performed.
 *
 *  This implementation can perform incremental EM by using multiple
 *  MaximizationSteps. An expectation step is performed between execution
 *  of each MaximizationStep. A call to iterate() will cycle through all
 *  MaximizationStep 's. A call to run() will call iterate() until the
 *  termination criteria have been met.
 *
 *  Having multiple and separate maximization steps allows for maximizing some
 *  parameters, performing another E-step, and then maximizing separate
 *  parameters, which may result in faster convergence in some cases.
 *
 *  \author Charles Vaske
 */
class EMAlg {
    private:
        /// All the data samples used during learning
        const Evidence &_evidence;

        /// How to do the expectation step
        InfAlg &_estep;

        /// The maximization steps to take
        std::vector<MaximizationStep> _msteps;

        /// Number of iterations done
        size_t _iters;

        /// History of likelihoods
        std::vector<Real> _lastLogZ;

        /// Maximum number of iterations
        size_t _max_iters;

        /// Convergence tolerance
        Real _log_z_tol;

    public:
        /// Key for setting maximum iterations
        static const std::string MAX_ITERS_KEY;
        /// Default maximum iterations
        static const size_t MAX_ITERS_DEFAULT;
        /// Key for setting likelihood termination condition
        static const std::string LOG_Z_TOL_KEY;
        /// Default likelihood tolerance
        static const Real LOG_Z_TOL_DEFAULT;

        /// Construct an EMAlg from several objects
        /** \param evidence Specifies the observed evidence
         *  \param estep Inference algorithm to be used for the E-step
         *  \param msteps Vector of maximization steps, each of which is a group of parameter estimation tasks
         *  \param termconditions Termination conditions @see setTermConditions()
         */
        EMAlg( const Evidence &evidence, InfAlg &estep, std::vector<MaximizationStep> &msteps, const PropertySet &termconditions )
          : _evidence(evidence), _estep(estep), _msteps(msteps), _iters(0), _lastLogZ(), _max_iters(MAX_ITERS_DEFAULT), _log_z_tol(LOG_Z_TOL_DEFAULT)
        {
              setTermConditions( termconditions );
        }

        /// Construct an EMAlg from Evidence \a evidence, an InfAlg \a estep, and an input stream \a mstep_file
        /** \see \ref fileformats-emalg
         */
        EMAlg( const Evidence &evidence, InfAlg &estep, std::istream &mstep_file );

        /// Change the conditions for termination
        /** There are two possible parameters in the PropertySet \a p:
         *    - \a max_iters maximum number of iterations
         *    - \a log_z_tol critical proportion of increase in logZ
         *
         *  \see hasSatisifiedTermConditions()
         */
        void setTermConditions( const PropertySet &p );

        /// Determine if the termination conditions have been met.
        /** There are two sufficient termination conditions:
         *    -# the maximum number of iterations has been performed
         *    -# the ratio of logZ increase over previous logZ is less than the
         *       tolerance, i.e.,
         *       \f$ \frac{\log(Z_t) - \log(Z_{t-1})}{| \log(Z_{t-1}) | } < \mathrm{tol} \f$.
         */
        bool hasSatisfiedTermConditions() const;

        /// Return the last calculated log likelihood
        Real logZ() const { return _lastLogZ.back(); }

        /// Returns number of iterations done so far
        size_t Iterations() const { return _iters; }

        /// Get the E-step method used
        const InfAlg& eStep() const { return _estep; }

        /// Iterate once over all maximization steps
        /** \return Log-likelihood after iteration
         */
        Real iterate();

        /// Iterate over a single MaximizationStep
        Real iterate( MaximizationStep &mstep );

        /// Iterate until termination conditions are satisfied
        void run();

    /// \name Iterator interface
    //@{
        /// Iterator over the maximization steps
        typedef std::vector<MaximizationStep>::iterator s_iterator;
        /// Constant iterator over the maximization steps
        typedef std::vector<MaximizationStep>::const_iterator const_s_iterator;

        /// Returns iterator that points to the first maximization step
        s_iterator s_begin() { return _msteps.begin(); }
        /// Returns constant iterator that points to the first maximization step
        const_s_iterator s_begin() const { return _msteps.begin(); }
        /// Returns iterator that points beyond the last maximization step
        s_iterator s_end() { return _msteps.end(); }
        /// Returns constant iterator that points beyond the last maximization step
        const_s_iterator s_end() const { return _msteps.end(); }
    //@}
};


} // end of namespace dai


/** \example example_sprinkler_em.cpp
 *  This example shows how to use the EMAlg class.
 */


#endif
