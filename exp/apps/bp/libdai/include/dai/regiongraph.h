/*  This file is part of libDAI - http://www.libdai.org/
 *
 *  Copyright (c) 2006-2011, The libDAI authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
 */


/// \file
/// \brief Defines classes Region, FRegion and RegionGraph, which implement a particular subclass of region graphs.


#ifndef __defined_libdai_regiongraph_h
#define __defined_libdai_regiongraph_h


#include <iostream>
#include <dai/bipgraph.h>
#include <dai/factorgraph.h>
#include <dai/weightedgraph.h>


namespace dai {


/// A Region is a set of variables with a counting number
class Region : public VarSet {
    private:
        /// Counting number
        Real _c;

    public:
        /// Default constructor
        Region() : VarSet(), _c(1.0) {}

        /// Construct from a set of variables and a counting number
        Region( const VarSet& x, Real c ) : VarSet(x), _c(c) {}

        /// Returns constant reference to counting number
        const Real& c() const { return _c; }

        /// Returns reference to counting number
        Real& c() { return _c; }
};


/// An FRegion is a factor with a counting number
class FRegion : public Factor {
    private:
        /// Counting number
        Real _c;

    public:
        /// Default constructor
        FRegion() : Factor(), _c(1.0) {}

        /// Constructs from a factor and a counting number
        FRegion( const Factor& x, Real c ) : Factor(x), _c(c) {}

        /// Returns constant reference to counting number
        const Real& c() const { return _c; }

        /// Returns reference to counting number
        Real& c() { return _c; }
};


/// A RegionGraph combines a bipartite graph consisting of outer regions (type FRegion) and inner regions (type Region) with a FactorGraph
/** A RegionGraph inherits from a FactorGraph and adds additional structure in the form of a "region graph". Our definition of region graph
 *  is inspired by [\ref HAK03], which is less general than the definition given in [\ref YFW05].
 *
 *  The extra structure described by a RegionGraph compared with that described by a FactorGraph is:
 *  - a set of outer regions (indexed by \f$\alpha\f$), where each outer region consists of
 *    - a factor defined on a subset of variables
 *    - a counting number
 *  - a set of inner regions (indexed by \f$\beta\f$), where each inner region consists of
 *    - a subset of variables
 *    - a counting number
 *  - edges between inner and outer regions
 *
 *  Each factor in the factor graph belongs to an outer region; normally, the factor contents
 *  of an outer region would be the product of all the factors that belong to that region.
 *  \idea Generalize the definition of region graphs to the one given in [\ref YFW05], i.e., replace
 *  the current implementation which uses a BipartiteGraph with one that uses a DAG.
 *  \idea The outer regions are products of factors; right now, this product is constantly cached:
 *  changing one factor results in an update of all relevant outer regions. This may not be the most
 *  efficient approach; an alternative would be to only precompute the factor products at the start
 *  of an inference algorithm - e.g., in init(). This has the additional advantage that FactorGraph
 e  can offer write access to its factors.
 */
class RegionGraph : public FactorGraph {
    protected:
        /// Stores the neighborhood structure
        BipartiteGraph          _G;

        /// The outer regions (corresponding to nodes of type 1)
        std::vector<FRegion>    _ORs;

        /// The inner regions (corresponding to nodes of type 2)
        std::vector<Region>     _IRs;

        /// Stores for each factor index the index of the outer region it belongs to
        std::vector<size_t>     _fac2OR;


    public:
    /// \name Constructors and destructors
    //@{
        /// Default constructor
        RegionGraph() : FactorGraph(), _G(), _ORs(), _IRs(), _fac2OR() {}

        /// Constructs a region graph from a factor graph, a vector of outer regions, a vector of inner regions and a vector of edges
        /** The counting numbers for the outer regions are set to 1.
         */
        RegionGraph( const FactorGraph& fg, const std::vector<VarSet>& ors, const std::vector<Region>& irs, const std::vector<std::pair<size_t,size_t> >& edges ) : FactorGraph(), _G(), _ORs(), _IRs(), _fac2OR() {
            construct( fg, ors, irs, edges );

            // Check counting numbers
#ifdef DAI_DEBUG
            checkCountingNumbers();
#endif
        }

        /// Constructs a region graph from a factor graph and a vector of outer clusters (CVM style)
        /** The region graph is constructed as in the Cluster Variation Method. 
         *
         *  The outer regions have as variable subsets the clusters specified in \a cl. 
         *  Each factor in the factor graph \a fg is assigned to one of the outer regions. 
         *  Each outer region gets counting number 1. 
         *
         *  The inner regions are (repeated) intersections of outer regions.
         *  An inner and an outer region are connected if the variables in the inner region form a
         *  subset of the variables in the outer region. The counting numbers for the inner
         *  regions are calculated by calcCountingNumbers() and satisfy the Moebius formula.
         */
        RegionGraph( const FactorGraph& fg, const std::vector<VarSet>& cl ) : FactorGraph(), _G(), _ORs(), _IRs(), _fac2OR() {
            constructCVM( fg, cl );

            // Check counting numbers
#ifdef DAI_DEBUG
            checkCountingNumbers();
#endif
        }

        /// Clone \c *this (virtual copy constructor)
        virtual RegionGraph* clone() const { return new RegionGraph(*this); }
    //@}

    /// \name Accessors and mutators
    //@{
        /// Returns number of outer regions
        size_t nrORs() const { return _ORs.size(); }
        /// Returns number of inner regions
        size_t nrIRs() const { return _IRs.size(); }

        /// Returns constant reference to outer region \a alpha
        const FRegion& OR( size_t alpha ) const {
            DAI_DEBASSERT( alpha < nrORs() );
            return _ORs[alpha]; 
        }
        /// Returns reference to outer region \a alpha
        FRegion& OR( size_t alpha ) {
            DAI_DEBASSERT( alpha < nrORs() );
            return _ORs[alpha]; 
        }

        /// Returns constant reference to inner region \a beta
        const Region& IR( size_t beta ) const {
            DAI_DEBASSERT( beta < nrIRs() );
            return _IRs[beta]; 
        }
        /// Returns reference to inner region \a beta
        Region& IR( size_t beta ) {
            DAI_DEBASSERT( beta < nrIRs() );
            return _IRs[beta];
        }

        /// Returns the index of the outer region to which the \a I 'th factor corresponds
        size_t fac2OR( size_t I ) const {
            DAI_DEBASSERT( I < nrFactors() );
            DAI_DEBASSERT( I < _fac2OR.size() );
            return _fac2OR[I];
        }

        /// Returns constant reference to the neighbors of outer region \a alpha
        const Neighbors& nbOR( size_t alpha ) const { return _G.nb1(alpha); }

        /// Returns constant reference to the neighbors of inner region \a beta
        const Neighbors& nbIR( size_t beta ) const { return _G.nb2(beta); }

        /// Returns DAG structure of the region graph
        /** \note Currently, the DAG is implemented as a BipartiteGraph; the nodes of
         *  type 1 are the outer regions, the nodes of type 2 the inner regions, and
         *  edges correspond with arrows from nodes of type 1 to type 2.
         */
        const BipartiteGraph& DAG() const { return _G; }
    //@}

    /// \name Queries
    //@{
        /// Check whether the counting numbers are valid
        /** Counting numbers are said to be (variable) valid if for each variable \f$x\f$,
         *    \f[\sum_{\alpha \ni x} c_\alpha + \sum_{\beta \ni x} c_\beta = 1\f]
         *  or in words, if the sum of the counting numbers of the regions
         *  that contain the variable equals one.
         */
        bool checkCountingNumbers() const;
    //@}

    /// \name Operations
    //@{
        /// Set the content of the \a I 'th factor and make a backup of its old content if \a backup == \c true
        virtual void setFactor( size_t I, const Factor& newFactor, bool backup = false ) {
            FactorGraph::setFactor( I, newFactor, backup );
            recomputeOR( I );
        }

        /// Set the contents of all factors as specified by \a facs and make a backup of the old contents if \a backup == \c true
        virtual void setFactors( const std::map<size_t, Factor>& facs, bool backup = false ) {
            FactorGraph::setFactors( facs, backup );
            VarSet ns;
            for( std::map<size_t, Factor>::const_iterator fac = facs.begin(); fac != facs.end(); fac++ )
                ns |= fac->second.vars();
            recomputeORs( ns );
        }
    //@}

    /// \name Input/output
    //@{
        /// Reads a region graph from a file
        /** \note Not implemented yet
         */
        virtual void ReadFromFile( const char* /*filename*/ ) {
            DAI_THROW(NOT_IMPLEMENTED);
        }

        /// Writes a factor graph to a file
        /** \note Not implemented yet
         */
        virtual void WriteToFile( const char* /*filename*/, size_t /*precision*/=15 ) const {
            DAI_THROW(NOT_IMPLEMENTED);
        }

        /// Writes a RegionGraph to an output stream
        friend std::ostream& operator<< ( std::ostream& os, const RegionGraph& rg );

        /// Writes a region graph to a GraphViz .dot file
        /** \note Not implemented yet
         */
        virtual void printDot( std::ostream& /*os*/ ) const {
            DAI_THROW(NOT_IMPLEMENTED);
        }
    //@}

    protected:
        /// Helper function for constructors
        void construct( const FactorGraph& fg, const std::vector<VarSet>& ors, const std::vector<Region>& irs, const std::vector<std::pair<size_t,size_t> >& edges );

        /// Helper function for constructors (CVM style)
        void constructCVM( const FactorGraph& fg, const std::vector<VarSet>& cl, size_t verbose=0 );

        /// Recompute all outer regions
        /** The factor contents of each outer region is set to the product of the factors belonging to that region.
         */
        void recomputeORs();

        /// Recompute all outer regions involving the variables in \a vs
        /** The factor contents of each outer region involving at least one of the variables in \a vs is set to the product of the factors belonging to that region.
         */
        void recomputeORs( const VarSet& vs );

        /// Recompute all outer regions involving factor \a I
        /** The factor contents of each outer region involving the \a I 'th factor is set to the product of the factors belonging to that region.
         */
        void recomputeOR( size_t I );

        /// Calculates counting numbers of inner regions based upon counting numbers of outer regions
        /** The counting numbers of the inner regions are set using the Moebius inversion formula:
         *    \f[ c_\beta := 1 - \sum_{\gamma \in \mathrm{an}(\beta)} c_\gamma \f]
         *  where \f$\mathrm{an}(\beta)\f$ are the ancestors of inner region \f$\beta\f$ according to
         *  the partial ordering induced by the subset relation (i.e., a region is a child of another
         *  region if its variables are a subset of the variables of its parent region).
         */
        void calcCVMCountingNumbers();

};


} // end of namespace dai


#endif
