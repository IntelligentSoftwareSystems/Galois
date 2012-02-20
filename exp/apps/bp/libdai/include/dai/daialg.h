/*  This file is part of libDAI - http://www.libdai.org/
 *
 *  Copyright (c) 2006-2011, The libDAI authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
 */


/// \file
/// \brief Defines the general interface for inference methods in libDAI (classes InfAlg, DaiAlg<>, DaiAlgFG and DaiAlgRG).


#ifndef __defined_libdai_daialg_h
#define __defined_libdai_daialg_h


#include <string>
#include <iostream>
#include <vector>
#include <dai/factorgraph.h>
#include <dai/regiongraph.h>
#include <dai/properties.h>


namespace dai {


/// InfAlg is an abstract base class, defining the common interface of all inference algorithms in libDAI.
/** \idea General marginalization functions like calcMarginal() now copy a complete InfAlg object. Instead,
 *  it would make more sense that they construct a new object without copying the FactorGraph or RegionGraph.
 *  Or they can simply be made methods of the general InfAlg class.
 *  \idea Use a PropertySet as output of an InfAlg, instead of functions like maxDiff() and Iterations().
 */
class InfAlg {
    public:
    /// \name Constructors/destructors
    //@{
        /// Virtual destructor (needed because this class contains virtual functions)
        virtual ~InfAlg() {}

        /// Returns a pointer to a new, cloned copy of \c *this (i.e., virtual copy constructor)
        virtual InfAlg* clone() const = 0;

        /// Returns a pointer to a newly constructed inference algorithm
        /** \param fg Factor graph on which to perform the inference algorithm;
         *  \param opts Parameters passed to constructor of inference algorithm;
         */
        virtual InfAlg* construct( const FactorGraph &fg, const PropertySet &opts ) const = 0;
    //@}

    /// \name Queries
    //@{
        /// Returns the name of the algorithm
        virtual std::string name() const = 0;

        /// Identifies itself for logging purposes
        virtual std::string identify() const {
            return name() + printProperties();
        }

        /// Returns reference to underlying FactorGraph.
        virtual FactorGraph &fg() = 0;

        /// Returns constant reference to underlying FactorGraph.
        virtual const FactorGraph &fg() const = 0;
    //@}

    /// \name Inference interface
    //@{
        /// Initializes all data structures of the approximate inference algorithm.
        /** \note This method should be called at least once before run() is called.
         */
        virtual void init() = 0;

        /// Initializes all data structures corresponding to some set of variables.
        /** This method can be used to do a partial initialization after a part of the factor graph has changed.
         *  Instead of initializing all data structures, it only initializes those involving the variables in \a vs.
         *  \throw NOT_IMPLEMENTED if not implemented/supported
         */
        virtual void init( const VarSet &vs ) = 0;

        /// Runs the approximate inference algorithm.
        /** \note Before run() is called the first time, init() should have been called.
         */
        virtual Real run() = 0;

        /// Returns the (approximate) marginal probability distribution of a variable.
        /** \note Before this method is called, run() should have been called.
         */
        virtual Factor belief( const Var &v ) const { return belief( VarSet(v) ); }

        /// Returns the (approximate) marginal probability distribution of a set of variables.
        /** \note Before this method is called, run() should have been called.
         *  \throw NOT_IMPLEMENTED if not implemented/supported.
         *  \throw BELIEF_NOT_AVAILABLE if the requested belief cannot be calculated with this algorithm.
         */
        virtual Factor belief( const VarSet &vs ) const = 0;

        /// Returns the (approximate) marginal probability distribution of the variable with index \a i.
        /** For some approximate inference algorithms, using beliefV() is preferred to belief() for performance reasons.
         *  \note Before this method is called, run() should have been called.
         */
        virtual Factor beliefV( size_t i ) const { return belief( fg().var(i) ); }

        /// Returns the (approximate) marginal probability distribution of the variables on which factor \a I depends.
        /** For some approximate inference algorithms, using beliefF() is preferred to belief() for performance reasons.
         *  \note Before this method is called, run() should have been called.
         */
        virtual Factor beliefF( size_t I ) const { return belief( fg().factor(I).vars() ); }

        /// Returns all beliefs (approximate marginal probability distributions) calculated by the algorithm.
        /** \note Before this method is called, run() should have been called.
         */
        virtual std::vector<Factor> beliefs() const = 0;

        /// Returns the logarithm of the (approximated) partition sum (normalizing constant of the factor graph).
        /** \note Before this method is called, run() should have been called.
         *  \throw NOT_IMPLEMENTED if not implemented/supported
         */
        virtual Real logZ() const = 0;

        /// Calculates the joint state of all variables that has maximum probability
        /** \note Before this method is called, run() should have been called.
         *  \throw NOT_IMPLEMENTED if not implemented/supported
         */
        virtual std::vector<std::size_t> findMaximum() const { DAI_THROW(NOT_IMPLEMENTED); }

        /// Returns maximum difference between single variable beliefs in the last iteration.
        /** \throw NOT_IMPLEMENTED if not implemented/supported
         */
        virtual Real maxDiff() const { DAI_THROW(NOT_IMPLEMENTED); };

        /// Returns number of iterations done (one iteration passes over the complete factorgraph).
        /** \throw NOT_IMPLEMENTED if not implemented/supported
         */
        virtual size_t Iterations() const { DAI_THROW(NOT_IMPLEMENTED); };

        /// Sets maximum number of iterations (one iteration passes over the complete factorgraph).
        /** \throw NOT_IMPLEMENTED if not implemented/supported
         */
        virtual void setMaxIter( size_t /*maxiter*/ ) { DAI_THROW(NOT_IMPLEMENTED); }
    //@}

    /// \name Changing the factor graph
    //@{
        /// Clamp variable with index \a i to value \a x (i.e. multiply with a Kronecker delta \f$\delta_{x_i, x}\f$)
        /** If \a backup == \c true, make a backup of all factors that are changed.
         */
        virtual void clamp( size_t i, size_t x, bool backup = false ) = 0;

        /// Sets all factors interacting with variable with index \a i to one.
        /** If \a backup == \c true, make a backup of all factors that are changed.
         */
        virtual void makeCavity( size_t i, bool backup = false ) = 0;
    //@}

    /// \name Backup/restore mechanism for factors
    //@{
        /// Make a backup copy of factor \a I
        /** \throw MULTIPLE_UNDO if a backup already exists
         */
        virtual void backupFactor( size_t I ) = 0;
        /// Make backup copies of all factors involving the variables in \a vs
        /** \throw MULTIPLE_UNDO if a backup already exists
         */
        virtual void backupFactors( const VarSet &vs ) = 0;

        /// Restore factor \a I from its backup copy
        virtual void restoreFactor( size_t I ) = 0;
        /// Restore the factors involving the variables in \a vs from their backup copies
        virtual void restoreFactors( const VarSet &vs ) = 0;
    //@}

    /// \name Managing parameters
    //@{
        /// Set parameters of this inference algorithm.
        /** The parameters are set according to the PropertySet \a opts. 
         *  The values can be stored either as std::string or as the type of the corresponding MF::props member.
         */
        virtual void setProperties( const PropertySet &opts ) = 0;
        /// Returns parameters of this inference algorithm converted into a PropertySet.
        virtual PropertySet getProperties() const = 0;
        /// Returns parameters of this inference algorithm formatted as a string in the format "[key1=val1,key2=val2,...,keyn=valn]".
        virtual std::string printProperties() const = 0;
    //@}
};


/// Combines the abstract base class InfAlg with a graphical model (e.g., a FactorGraph or RegionGraph).
/** Inference algorithms in libDAI directly inherit from a DAIAlg, currently either
 *  from a DAIAlg<FactorGraph> or from a DAIAlg<RegionGraph>.
 *
 *  \tparam GRM Should be castable to FactorGraph
 *  \idea A DAIAlg should not inherit from a FactorGraph or RegionGraph, but should
 *  store a reference to the graphical model object. This prevents needless copying
 *  of (possibly large) data structures. Disadvantage: the caller must not change
 *  the graphical model between calls to the inference algorithm (maybe a smart_ptr
 *  or some locking mechanism would help here?).
 */
template <class GRM>
class DAIAlg : public InfAlg, public GRM {
    public:
    /// \name Constructors/destructors
    //@{
        /// Default constructor
        DAIAlg() : InfAlg(), GRM() {}

        /// Construct from GRM
        DAIAlg( const GRM &grm ) : InfAlg(), GRM(grm) {}
    //@}

    /// \name Queries
    //@{
        /// Returns reference to underlying FactorGraph.
        FactorGraph &fg() { return (FactorGraph &)(*this); }

        /// Returns constant reference to underlying FactorGraph.
        const FactorGraph &fg() const { return (const FactorGraph &)(*this); }
    //@}

    /// \name Changing the factor graph
    //@{
        /// Clamp variable with index \a i to value \a x (i.e. multiply with a Kronecker delta \f$\delta_{x_i, x}\f$)
        /** If \a backup == \c true, make a backup of all factors that are changed.
         */
        void clamp( size_t i, size_t x, bool backup = false ) { GRM::clamp( i, x, backup ); }

        /// Sets all factors interacting with variable with index \a i to one.
        /** If \a backup == \c true, make a backup of all factors that are changed.
         */
        void makeCavity( size_t i, bool backup = false ) { GRM::makeCavity( i, backup ); }
    //@}

    /// \name Backup/restore mechanism for factors
    //@{
        /// Make a backup copy of factor \a I
        void backupFactor( size_t I ) { GRM::backupFactor( I ); }
        /// Make backup copies of all factors involving the variables in \a vs
        void backupFactors( const VarSet &vs ) { GRM::backupFactors( vs ); }

        /// Restore factor \a I from its backup copy
        void restoreFactor( size_t I ) { GRM::restoreFactor( I ); }
        /// Restore the factors involving the variables in \a vs from their backup copies
        void restoreFactors( const VarSet &vs ) { GRM::restoreFactors( vs ); }
    //@}
};


/// Base class for inference algorithms that operate on a FactorGraph
typedef DAIAlg<FactorGraph> DAIAlgFG;

/// Base class for inference algorithms that operate on a RegionGraph
typedef DAIAlg<RegionGraph> DAIAlgRG;


/// Calculates the marginal probability distribution for \a vs using inference algorithm \a obj.
/** calcMarginal() works by clamping all variables in \a vs and calculating the partition sum for each clamped state.
 *  Therefore, it can be used in combination with any inference algorithm that can calculate/approximate partition sums.
 *  \param obj instance of inference algorithm to be used 
 *  \param vs variables for which the marginal should be calculated
 *  \param reInit should be set to \c true if at least one of the possible clamped states would be invalid (leading to a factor graph with zero partition sum).
 */
Factor calcMarginal( const InfAlg& obj, const VarSet& vs, bool reInit );


/// Calculates beliefs for all pairs of variables in \a vs using inference algorithm \a obj.
/** calcPairBeliefs() works by 
 *  - clamping single variables in \a vs and calculating the partition sum and the single variable beliefs for each clamped state, if \a accurate == \c false;
 *  - clamping pairs of variables in \a vs and calculating the partition sum for each clamped state, if \a accurate == \c true.
 *
 *  Therefore, it can be used in combination with any inference algorithm that can calculate/approximate partition sums (and single variable beliefs, if
 *  \a accurate == \c true).
 *  \param obj instance of inference algorithm to be used 
 *  \param vs variables for which the pair beliefs should be calculated
 *  \param reInit should be set to \c true if at least one of the possible clamped states would be invalid (leading to a factor graph with zero partition sum).
 *  \param accurate if \c true, uses a slower but more accurate approximation algorithm
 */
std::vector<Factor> calcPairBeliefs( const InfAlg& obj, const VarSet& vs, bool reInit, bool accurate=false );


/// Calculates the joint state of all variables that has maximum probability, according to the inference algorithm \a obj
/** \note Before this method is called, obj.run() should have been called.
 */
std::vector<size_t> findMaximum( const InfAlg& obj );


} // end of namespace dai


#endif
