/*  This file is part of libDAI - http://www.libdai.org/
 *
 *  Copyright (c) 2006-2011, The libDAI authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
 */


/// \file
/// \brief Defines the FactorGraph class, which represents factor graphs (e.g., Bayesian networks or Markov random fields)


#ifndef __defined_libdai_factorgraph_h
#define __defined_libdai_factorgraph_h


#include <iostream>
#include <map>
#include <dai/bipgraph.h>
#include <dai/graph.h>
#include <dai/factor.h>


namespace dai {


/// Represents a factor graph.
/** Both Bayesian Networks and Markov random fields can be represented in a
 *  unifying representation, called <em>factor graph</em> [\ref KFL01],
 *  implemented in libDAI by the FactorGraph class.
 *  
 *  Consider a probability distribution over \f$N\f$ discrete random variables
 *  \f$x_0,x_1,\dots,x_{N-1}\f$ that factorizes as a product of \f$M\f$ factors, each of
 *  which depends on some subset of the variables:
 *  \f[
 *    P(x_0,x_1,\dots,x_{N-1}) = \frac{1}{Z} \prod_{I=0}^{M-1} f_I(x_I), \qquad
 *    Z = \sum_{x_0}\dots\sum_{x_{N-1}} \prod_{I=0}^{M-1} f_I(X_I).
 *  \f]
 *  Each factor \f$f_I\f$ is a function from an associated subset
 *  of variables \f$X_I \subset \{x_0,x_1,\dots,x_{N-1}\}\f$ to the nonnegative
 *  real numbers.
 * 
 *  For a Bayesian network, each factor corresponds to a (conditional)
 *  probability table, whereas for a Markov random field, each factor
 *  corresponds to a maximal clique of the undirected graph.
 *
 *  Factor graphs explicitly express the factorization structure of the
 *  corresponding probability distribution. A factor graph is a bipartite graph,
 *  containing variable nodes and factor nodes, and an edge between a variable
 *  node and a factor node if the corresponding factor depends on that variable.
 *  In libDAI, this structure is represented by a BipartiteGraph. 
 *
 *  So basically, a FactorGraph consists of a BipartiteGraph, a vector of Var 's
 *  and a vector of TFactor 's.
 *
 *  \idea Alternative implementation of undo factor changes: the only things that have to be
 *  undone currently are setting a factor to 1 and setting a factor to a Kronecker delta. This
 *  could also be implemented in the TFactor itself, which could maintain its state
 *  (ones/delta/full) and act accordingly. Update: it seems that the proposed functionality 
 *  would not be enough for CBP, for which it would make more sense to add more levels of
 *  backup/restore.
 *
 *  \todo Write a method that applies evidence (should we represent evidence as a map<Var,size_t> or as a map<size_t,size_t>?)
 */ 
class FactorGraph {
    private:
        /// Stores the neighborhood structure
        BipartiteGraph           _G;
        /// Stores the variables
        std::vector<Var>         _vars;
        /// Stores the factors
        std::vector<Factor>      _factors;
        /// Stores backups of some factors
        std::map<size_t,Factor>  _backup;

    public:
    /// \name Constructors and destructors
    //@{
        /// Default constructor
        FactorGraph() : _G(), _vars(), _factors(), _backup() {}

        /// Constructs a factor graph from a vector of factors
        FactorGraph( const std::vector<Factor>& P );

        /// Constructs a factor graph from given factor and variable iterators
        /** \tparam FactorInputIterator Iterates over instances of type dai::Factor
         *  \tparam VarInputIterator Iterates over instances of type Var
         *  \pre Assumes that the set of variables in [\a varBegin, \a varEnd) is the union of the variables in the factors in [\a facBegin, \a facEnd)
         */
        template<typename FactorInputIterator, typename VarInputIterator>
        FactorGraph(FactorInputIterator facBegin, FactorInputIterator facEnd, VarInputIterator varBegin, VarInputIterator varEnd, size_t nrFacHint = 0, size_t nrVarHint = 0 );

        /// Destructor
        virtual ~FactorGraph() {}

        /// Virtual copy constructor
        virtual FactorGraph* clone() const { return new FactorGraph(*this); }
    //@}

    /// \name Accessors and mutators
    //@{ 
        /// Returns constant reference the \a i 'th variable
        const Var& var( size_t i ) const { 
            DAI_DEBASSERT( i < nrVars() );
            return _vars[i]; 
        }

        /// Returns constant reference to all variables
        const std::vector<Var>& vars() const { return _vars; }

        /// Returns constant reference to \a I 'th factor
        const Factor& factor( size_t I ) const { 
            DAI_DEBASSERT( I < nrFactors() );
            return _factors[I]; 
        }
        /// Returns constant reference to all factors
        const std::vector<Factor>& factors() const { return _factors; }

        /// Returns constant reference to neighbors of the \a i 'th variable
        const Neighbors& nbV( size_t i ) const { return _G.nb1(i); }
        /// Returns constant reference to neighbors of the \a I 'th factor
        const Neighbors& nbF( size_t I ) const { return _G.nb2(I); }
        /// Returns constant reference to the \a _I 'th neighbor of the \a i 'th variable
        const Neighbor& nbV( size_t i, size_t _I ) const { return _G.nb1(i)[_I]; }
        /// Returns constant reference to the \a _i 'th neighbor of the \a I 'th factor
        const Neighbor& nbF( size_t I, size_t _i ) const { return _G.nb2(I)[_i]; }
    //@}

    /// \name Queries
    //@{
        /// Returns neighborhood structure
        const BipartiteGraph& bipGraph() const { return _G; }
        /// Returns number of variables
        size_t nrVars() const { return vars().size(); }
        /// Returns number of factors
        size_t nrFactors() const { return factors().size(); }
        /// Calculates number of edges
        /** \note Time complexity: O(nrVars())
         */
        size_t nrEdges() const { return _G.nrEdges(); }

        /// Returns the index of a particular variable
        /** \note Time complexity: O(nrVars())
         *  \throw OBJECT_NOT_FOUND if the variable is not part of this factor graph
         */
        size_t findVar( const Var& n ) const {
            size_t i = find( vars().begin(), vars().end(), n ) - vars().begin();
            if( i == nrVars() )
                DAI_THROW(OBJECT_NOT_FOUND);
            return i;
        }

        /// Returns a set of indexes corresponding to a set of variables
        /** \note Time complexity: O( nrVars() * ns.size() )
         *  \throw OBJECT_NOT_FOUND if one of the variables is not part of this factor graph
         */
        SmallSet<size_t> findVars( const VarSet& ns ) const {
            SmallSet<size_t> result;
            for( VarSet::const_iterator n = ns.begin(); n != ns.end(); n++ )
                result.insert( findVar( *n ) );
            return result;
        }

        /// Returns index of the first factor that depends on the variables
        /** \note Time complexity: O(nrFactors())
         *  \throw OBJECT_NOT_FOUND if no factor in this factor graph depends on those variables
         */
        size_t findFactor( const VarSet& ns ) const {
            size_t I;
            for( I = 0; I < nrFactors(); I++ )
                if( factor(I).vars() == ns )
                    break;
            if( I == nrFactors() )
                DAI_THROW(OBJECT_NOT_FOUND);
            return I;
        }

        /// Return all variables that occur in a factor involving the \a i 'th variable, itself included
        VarSet Delta( size_t i ) const;

        /// Return all variables that occur in a factor involving some variable in \a vs, \a vs itself included
        VarSet Delta( const VarSet& vs ) const;

        /// Return all variables that occur in a factor involving the \a i 'th variable, itself excluded
        VarSet delta( size_t i ) const {
            return( Delta( i ) / var( i ) );
        }

        /// Return all variables that occur in a factor involving some variable in \a vs, \a vs itself excluded
        VarSet delta( const VarSet& vs ) const {
            return Delta( vs ) / vs;
        }

        /// Returns \c true if the factor graph is connected
        bool isConnected() const { return _G.isConnected(); }

        /// Returns \c true if the factor graph is a tree (i.e., has no cycles and is connected)
        bool isTree() const { return _G.isTree(); }

        /// Returns \c true if each factor depends on at most two variables
        bool isPairwise() const;

        /// Returns \c true if each variable has only two possible values
        bool isBinary() const;

        /// Constructs the corresponding Markov graph 
        /** \note The Markov graph has the variables as nodes and an edge
         *  between two variables if and only if the variables share a factor.
         */
        GraphAL MarkovGraph() const;

        /// Returns whether the \a I 'th factor is maximal
        /** \note A factor (domain) is \a maximal if and only if it is not a
         *  strict subset of another factor domain.
         */
        bool isMaximal( size_t I ) const;

        /// Returns the index of a maximal factor that contains the \a I 'th factor
        /** \note A factor (domain) is \a maximal if and only if it is not a
         *  strict subset of another factor domain.
         */
        size_t maximalFactor( size_t I ) const;

        /// Returns the maximal factor domains in this factorgraph
        /** \note A factor domain is \a maximal if and only if it is not a
         *  strict subset of another factor domain.
         */
        std::vector<VarSet> maximalFactorDomains() const;

        /// Evaluates the log score (i.e., minus the energy) of the joint configuration \a statevec
        Real logScore( const std::vector<size_t>& statevec ) const;
    //@}

    /// \name Backup/restore mechanism for factors
    //@{
        /// Set the content of the \a I 'th factor and make a backup of its old content if \a backup == \c true
        virtual void setFactor( size_t I, const Factor& newFactor, bool backup = false ) {
            DAI_ASSERT( newFactor.vars() == factor(I).vars() );
            if( backup )
                backupFactor( I );
            _factors[I] = newFactor;
        }

        /// Set the contents of all factors as specified by \a facs and make a backup of the old contents if \a backup == \c true
        virtual void setFactors( const std::map<size_t, Factor>& facs, bool backup = false ) {
            for( std::map<size_t, Factor>::const_iterator fac = facs.begin(); fac != facs.end(); fac++ ) {
                if( backup )
                    backupFactor( fac->first );
                setFactor( fac->first, fac->second );
            }
        }

        /// Makes a backup of the \a I 'th factor
        /** \throw MULTIPLE_UNDO if a backup already exists
         */
        void backupFactor( size_t I );

        /// Restores the \a I 'th factor from the backup (it should be backed up first)
        /** \throw OBJECT_NOT_FOUND if a backup does not exist
         */
        void restoreFactor( size_t I );

        /// Backup the factors specified by indices in \a facs
        /** \throw MULTIPLE_UNDO if a backup already exists
         */
        virtual void backupFactors( const std::set<size_t>& facs );

        /// Restore all factors to the backup copies
        virtual void restoreFactors();

        /// Makes a backup of all factors connected to a set of variables
        /** \throw MULTIPLE_UNDO if a backup already exists
         */
        void backupFactors( const VarSet& ns );

        /// Restores all factors connected to a set of variables from their backups
        void restoreFactors( const VarSet& ns );
    //@}

    /// \name Transformations
    //@{
        /// Returns a copy of \c *this, where all factors that are subsumed by some larger factor are merged with the larger factors.
        FactorGraph maximalFactors() const;

        /// Returns a copy of \c *this, where the \a i 'th variable has been clamped to value \a x
        /** \note This version changes the factor graph structure and thus returns a newly constructed FactorGraph
         *  and keeps the current one constant, contrary to clamp()
         */
        FactorGraph clamped( size_t i, size_t x ) const;
    //@}

    /// \name Operations
    //@{
        /// Clamp the \a i 'th variable to value \a x (i.e. multiply with a Kronecker delta \f$\delta_{x_i, x}\f$)
        /** If \a backup == \c true, make a backup of all factors that are changed
         */
        virtual void clamp( size_t i, size_t x, bool backup = false );

        /// Clamp a variable in a factor graph to have one out of a list of values
        /** If \a backup == \c true, make a backup of all factors that are changed
         */
        void clampVar( size_t i, const std::vector<size_t>& xis, bool backup = false );

        /// Clamp a factor in a factor graph to have one out of a list of values
        /** If \a backup == \c true, make a backup of all factors that are changed
         */
        void clampFactor( size_t I, const std::vector<size_t>& xIs, bool backup = false );

        /// Set all factors interacting with the \a i 'th variable to 1
        /** If \a backup == \c true, make a backup of all factors that are changed
         */
        virtual void makeCavity( size_t i, bool backup = false );
    //@}

    /// \name Input/Output
    //@{
        /// Reads a factor graph from a file
        /** \see \ref fileformats-factorgraph
         *  \throw CANNOT_READ_FILE if the file cannot be opened
         *  \throw INVALID_FACTORGRAPH_FILE if the file is not valid
         */
        virtual void ReadFromFile( const char *filename );

        /// Writes a factor graph to a file
        /** \see \ref fileformats-factorgraph
         *  \throw CANNOT_WRITE_FILE if the file cannot be written
         */
        virtual void WriteToFile( const char *filename, size_t precision=15 ) const;

        /// Writes a factor graph to an output stream
        /** \see \ref fileformats-factorgraph
         */
        friend std::ostream& operator<< (std::ostream& os, const FactorGraph& fg );

        /// Reads a factor graph from an input stream
        /** \see \ref fileformats-factorgraph
         *  \throw INVALID_FACTORGRAPH_FILE if the input stream is not valid
         */
        friend std::istream& operator>> (std::istream& is, FactorGraph& fg );

        /// Writes a factor graph to a GraphViz .dot file
        virtual void printDot( std::ostream& os ) const;
    //@}

    private:
        /// Part of constructors (creates edges, neighbors and adjacency matrix)
        void constructGraph( size_t nrEdges );
};


template<typename FactorInputIterator, typename VarInputIterator>
FactorGraph::FactorGraph(FactorInputIterator facBegin, FactorInputIterator facEnd, VarInputIterator varBegin, VarInputIterator varEnd, size_t nrFacHint, size_t nrVarHint ) : _G(), _backup() {
    // add factors
    size_t nrEdges = 0;
    _factors.reserve( nrFacHint );
    for( FactorInputIterator p2 = facBegin; p2 != facEnd; ++p2 ) {
        _factors.push_back( *p2 );
        nrEdges += p2->vars().size();
    }

    // add variables
    _vars.reserve( nrVarHint );
    for( VarInputIterator p1 = varBegin; p1 != varEnd; ++p1 )
        _vars.push_back( *p1 );

    // create graph structure
    constructGraph( nrEdges );
}


/** \example example_sprinkler.cpp
 *  This example illustrates how to manually construct a factor graph and
 *  write it to a file.
 */


} // end of namespace dai


#endif
