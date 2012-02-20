/*  This file is part of libDAI - http://www.libdai.org/
 *
 *  Copyright (c) 2006-2011, The libDAI authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
 */


/// \file
/// \brief Defines class ClusterGraph, which is used by JTree, TreeEP and HAK
/// \todo The "MinFill" and "WeightedMinFill" variable elimination heuristics seem designed for Markov graphs;
/// add similar heuristics which are designed for factor graphs.


#ifndef __defined_libdai_clustergraph_h
#define __defined_libdai_clustergraph_h


#include <set>
#include <vector>
#include <dai/varset.h>
#include <dai/bipgraph.h>
#include <dai/factorgraph.h>


namespace dai {


    /// A ClusterGraph is a hypergraph with variables as nodes, and "clusters" (sets of variables) as hyperedges.
    /** It is implemented as a bipartite graph with variable (Var) nodes and cluster (VarSet) nodes.
     *  One may think of a ClusterGraph as a FactorGraph without the actual factor values.
     *  \todo Remove the _vars and _clusters variables and use only the graph and a contextual factor graph.
     */
    class ClusterGraph {
        private:
            /// Stores the neighborhood structure
            BipartiteGraph       _G;

            /// Stores the variables corresponding to the nodes
            std::vector<Var>     _vars;

            /// Stores the clusters corresponding to the hyperedges
            std::vector<VarSet>  _clusters;

        public:
        /// \name Constructors and destructors
        //@{
            /// Default constructor
            ClusterGraph() : _G(), _vars(), _clusters() {}

            /// Construct from vector of VarSet 's
            ClusterGraph( const std::vector<VarSet>& cls );

            /// Construct from a factor graph
            /** Creates cluster graph which has factors in \a fg as clusters if \a onlyMaximal == \c false,
             *  and only the maximal factors in \a fg if \a onlyMaximal == \c true.
             */
            ClusterGraph( const FactorGraph& fg, bool onlyMaximal );
        //@}

        /// \name Queries
        //@{
            /// Returns a constant reference to the graph structure
            const BipartiteGraph& bipGraph() const { return _G; }

            /// Returns number of variables
            size_t nrVars() const { return _vars.size(); }

            /// Returns a constant reference to the variables
            const std::vector<Var>& vars() const { return _vars; }

            /// Returns a constant reference to the \a i'th variable
            const Var& var( size_t i ) const {
                DAI_DEBASSERT( i < nrVars() );
                return _vars[i]; 
            }

            /// Returns number of clusters
            size_t nrClusters() const { return _clusters.size(); }

            /// Returns a constant reference to the clusters
            const std::vector<VarSet>& clusters() const { return _clusters; }

            /// Returns a constant reference to the \a I'th cluster
            const VarSet& cluster( size_t I ) const {
                DAI_DEBASSERT( I < nrClusters() );
                return _clusters[I]; 
            }

            /// Returns the index of variable \a n
            size_t findVar( const Var& n ) const {
                return find( _vars.begin(), _vars.end(), n ) - _vars.begin();
            }

            /// Returns the index of a cluster \a cl
            size_t findCluster( const VarSet& cl ) const {
                return find( _clusters.begin(), _clusters.end(), cl ) - _clusters.begin();
            }

            /// Returns union of clusters that contain the \a i 'th variable
            VarSet Delta( size_t i ) const {
                VarSet result;
                foreach( const Neighbor& I, _G.nb1(i) )
                    result |= _clusters[I];
                return result;
            }

            /// Returns union of clusters that contain the \a i 'th (except this variable itself)
            VarSet delta( size_t i ) const {
                return Delta( i ) / _vars[i];
            }

            /// Returns \c true if variables with indices \a i1 and \a i2 are adjacent, i.e., both contained in the same cluster
            bool adj( size_t i1, size_t i2 ) const {
                if( i1 == i2 )
                    return false;
                bool result = false;
                foreach( const Neighbor& I, _G.nb1(i1) )
                    if( find( _G.nb2(I).begin(), _G.nb2(I).end(), i2 ) != _G.nb2(I).end() ) {
                        result = true;
                        break;
                    }
                return result;
            }

            /// Returns \c true if cluster \a I is not contained in a larger cluster
            bool isMaximal( size_t I ) const {
                DAI_DEBASSERT( I < _G.nrNodes2() );
                const VarSet & clI = _clusters[I];
                bool maximal = true;
                // The following may not be optimal, since it may repeatedly test the same cluster *J
                foreach( const Neighbor& i, _G.nb2(I) ) {
                    foreach( const Neighbor& J, _G.nb1(i) )
                        if( (J != I) && (clI << _clusters[J]) ) {
                            maximal = false;
                            break;
                        }
                    if( !maximal )
                        break;
                }
                return maximal;
            }
        //@}

        /// \name Operations
        //@{
            /// Inserts a cluster (if it does not already exist) and creates new variables, if necessary
            /** \note This function could be better optimized if the index of one variable in \a cl would be known.
             *        If one could assume _vars to be ordered, a binary search could be used instead of a linear one.
             */
            size_t insert( const VarSet& cl ) {
                size_t index = findCluster( cl );  // OPTIMIZE ME
                if( index == _clusters.size() ) {
                    _clusters.push_back( cl );
                    // add variables (if necessary) and calculate neighborhood of new cluster
                    std::vector<size_t> nbs;
                    for( VarSet::const_iterator n = cl.begin(); n != cl.end(); n++ ) {
                        size_t iter = findVar( *n );  // OPTIMIZE ME
                        nbs.push_back( iter );
                        if( iter == _vars.size() ) {
                            _G.addNode1();
                            _vars.push_back( *n );
                        }
                    }
                    _G.addNode2( nbs.begin(), nbs.end(), nbs.size() );
                }
                return index;
            }

            /// Erases all clusters that are not maximal
            ClusterGraph& eraseNonMaximal() {
                for( size_t I = 0; I < _G.nrNodes2(); ) {
                    if( !isMaximal(I) ) {
                        _clusters.erase( _clusters.begin() + I );
                        _G.eraseNode2(I);
                    } else
                        I++;
                }
                return *this;
            }

            /// Erases all clusters that contain the \a i 'th variable
            ClusterGraph& eraseSubsuming( size_t i ) {
                DAI_ASSERT( i < nrVars() );
                while( _G.nb1(i).size() ) {
                    _clusters.erase( _clusters.begin() + _G.nb1(i)[0] );
                    _G.eraseNode2( _G.nb1(i)[0] );
                }
                return *this;
            }

            /// Eliminates variable with index \a i, without deleting the variable itself
            /** \note This function can be better optimized
             */
            VarSet elimVar( size_t i ) {
                DAI_ASSERT( i < nrVars() );
                VarSet Di = Delta( i );
                insert( Di / var(i) );
                eraseSubsuming( i );
                eraseNonMaximal();
                return Di;
            }
        //@}

        /// \name Input/Ouput
        //@{
            /// Writes a ClusterGraph to an output stream
            friend std::ostream& operator << ( std::ostream& os, const ClusterGraph& cl ) {
                os << cl.clusters();
                return os;
            }
        //@}

        /// \name Variable elimination
        //@{
            /// Performs Variable Elimination, keeping track of the interactions that are created along the way.
            /** \tparam EliminationChoice should support "size_t operator()( const ClusterGraph &cl, const std::set<size_t> &remainingVars )"
             *  \param f function object which returns the next variable index to eliminate; for example, a dai::greedyVariableElimination object.
             *  \param maxStates maximum total number of states of all clusters in the output cluster graph (0 means no limit).
             *  \throws OUT_OF_MEMORY if total number of states becomes larger than maxStates
             *  \return A set of elimination "cliques".
             */
            template<class EliminationChoice>
            ClusterGraph VarElim( EliminationChoice f, size_t maxStates=0 ) const {
                // Make a copy
                ClusterGraph cl(*this);
                cl.eraseNonMaximal();

                ClusterGraph result;

                // Construct set of variable indices
                std::set<size_t> varindices;
                for( size_t i = 0; i < _vars.size(); ++i )
                    varindices.insert( i );

                // Do variable elimination
                BigInt totalStates = 0;
                while( !varindices.empty() ) {
                    size_t i = f( cl, varindices );
                    VarSet Di = cl.elimVar( i );
                    result.insert( Di );
                    if( maxStates ) {
                        totalStates += Di.nrStates();
                        if( totalStates > maxStates )
                            DAI_THROW(OUT_OF_MEMORY);
                    }
                    varindices.erase( i );
                }

                return result;
            }
        //@}
    };


    /// Helper object for dai::ClusterGraph::VarElim()
    /** Chooses the next variable to eliminate by picking them sequentially from a given vector of variables.
     */
    class sequentialVariableElimination {
        private:
            /// The variable elimination sequence
            std::vector<Var> seq;
            /// Counter
            size_t i;

        public:
            /// Construct from vector of variables
            sequentialVariableElimination( const std::vector<Var> s ) : seq(s), i(0) {}

            /// Returns next variable in sequence
           size_t operator()( const ClusterGraph &cl, const std::set<size_t> &/*remainingVars*/ );
    };


    /// Helper object for dai::ClusterGraph::VarElim()
    /** Chooses the next variable to eliminate greedily by taking the one that minimizes
     *  a given heuristic cost function.
     */
    class greedyVariableElimination {
        public:
            /// Type of cost functions to be used for greedy variable elimination
            typedef size_t (*eliminationCostFunction)(const ClusterGraph &, size_t);

        private:
            /// Pointer to the cost function used
            eliminationCostFunction heuristic;

        public:
            /// Construct from cost function
            /** \note Examples of cost functions are eliminationCost_MinFill() and eliminationCost_WeightedMinFill().
             */
            greedyVariableElimination( eliminationCostFunction h ) : heuristic(h) {}

            /// Returns the best variable from \a remainingVars to eliminate in the cluster graph \a cl by greedily minimizing the cost function.
            /** This function calculates the cost for eliminating each variable in \a remaingVars and returns the variable which has lowest cost.
             */
            size_t operator()( const ClusterGraph &cl, const std::set<size_t>& remainingVars );
    };


    /// Calculates cost of eliminating the \a i 'th variable from cluster graph \a cl according to the "MinNeighbors" criterion.
    /** The cost is measured as "number of neigboring nodes in the current adjacency graph",
     *  where the adjacency graph has the variables as its nodes and connects
     *  nodes \a i1 and \a i2 iff \a i1 and \a i2 occur together in some common cluster.
     */
    size_t eliminationCost_MinNeighbors( const ClusterGraph& cl, size_t i );


    /// Calculates cost of eliminating the \a i 'th variable from cluster graph \a cl according to the "MinWeight" criterion.
    /** The cost is measured as "product of weights of neighboring nodes in the current adjacency graph",
     *  where the adjacency graph has the variables as its nodes and connects
     *  nodes \a i1 and \a i2 iff \a i1 and \a i2 occur together in some common cluster.
     *  The weight of a node is the number of states of the corresponding variable.
     */
    size_t eliminationCost_MinWeight( const ClusterGraph& cl, size_t i );


    /// Calculates cost of eliminating the \a i 'th variable from cluster graph \a cl according to the "MinFill" criterion.
    /** The cost is measured as "number of added edges in the adjacency graph",
     *  where the adjacency graph has the variables as its nodes and connects
     *  nodes \a i1 and \a i2 iff \a i1 and \a i2 occur together in some common cluster.
     */
    size_t eliminationCost_MinFill( const ClusterGraph& cl, size_t i );


    /// Calculates cost of eliminating the \a i 'th variable from cluster graph \a cl according to the "WeightedMinFill" criterion.
    /** The cost is measured as "total weight of added edges in the adjacency graph",
     *  where the adjacency graph has the variables as its nodes and connects
     *  nodes \a i1 and \a i2 iff \a i1 and \a i2 occur together in some common cluster.
     *  The weight of an edge is the product of the number of states of the variables corresponding with its nodes.
     */
    size_t eliminationCost_WeightedMinFill( const ClusterGraph& cl, size_t i );


} // end of namespace dai


#endif
