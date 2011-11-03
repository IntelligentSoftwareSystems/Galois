/*  This file is part of libDAI - http://www.libdai.org/
 *
 *  Copyright (c) 2006-2011, The libDAI authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
 */


/// \file
/// \brief Defines the BipartiteGraph class, which represents a bipartite graph


#ifndef __defined_libdai_bipgraph_h
#define __defined_libdai_bipgraph_h


#include <ostream>
#include <vector>
#include <algorithm>
#include <dai/util.h>
#include <dai/smallset.h>
#include <dai/exceptions.h>
#include <dai/graph.h>


namespace dai {


/// Represents the neighborhood structure of nodes in an undirected, bipartite graph.
/** A bipartite graph has two types of nodes: type 1 and type 2. Edges can occur only between
 *  nodes of different type. Nodes are indexed by an unsigned integer. If there are nrNodes1()
 *  nodes of type 1 and nrNodes2() nodes of type 2, the nodes of type 1 are numbered
 *  0,1,2,...,nrNodes1()-1 and the nodes of type 2 are numbered 0,1,2,...,nrNodes2()-1. An edge
 *  between node \a n1 of type 1 and node \a n2 of type 2 is represented by a Edge(\a n1,\a n2).
 *
 *  A BipartiteGraph is implemented as a sparse adjacency list, i.e., it stores for each node a list of
 *  its neighboring nodes. More precisely: it stores for each node of type 1 a vector of Neighbor structures
 *  (accessible by the nb1() method) describing the neighboring nodes of type 2; similarly, for each node
 *  of type 2 it stores a vector of Neighbor structures (accessibly by the nb2() method) describing the
 *  neighboring nodes of type 1.
 *  Thus, each node has an associated variable of type BipartiteGraph::Neighbors, which is a vector of
 *  Neighbor structures, describing its neighboring nodes of the other type.
 *  \idea Cache second-order neighborhoods in BipartiteGraph.
 */
class BipartiteGraph {
    private:
        /// Contains for each node of type 1 a vector of its neighbors
        std::vector<Neighbors> _nb1;

        /// Contains for each node of type 2 a vector of its neighbors
        std::vector<Neighbors> _nb2;

        /// Used internally by isTree()
        struct levelType {
            /// Indices of nodes of type 1
            std::vector<size_t> ind1;       
            /// Indices of nodes of type 2
            std::vector<size_t> ind2;
        };

    public:
    /// \name Constructors and destructors
    //@{
        /// Default constructor (creates an empty bipartite graph)
        BipartiteGraph() : _nb1(), _nb2() {}

        /// Constructs BipartiteGraph with \a nr1 nodes of type 1, \a nr2 nodes of type 2 and no edges.
        BipartiteGraph( size_t nr1, size_t nr2 ) : _nb1(nr1), _nb2(nr2) {}

        /// Constructs BipartiteGraph from a range of edges.
        /** \tparam EdgeInputIterator Iterator that iterates over instances of Edge.
         *  \param nrNodes1 The number of nodes of type 1.
         *  \param nrNodes2 The number of nodes of type 2.
         *  \param begin Points to the first edge.
         *  \param end Points just beyond the last edge.
         *  \param check Whether to only add an edge if it does not exist already.
         */
        template<typename EdgeInputIterator>
        BipartiteGraph( size_t nrNodes1, size_t nrNodes2, EdgeInputIterator begin, EdgeInputIterator end, bool check=true ) : _nb1(), _nb2() {
            construct( nrNodes1, nrNodes2, begin, end, check );
        }
    //@}

    /// \name Accessors and mutators
    //@{
        /// Returns constant reference to the \a _i2 'th neighbor of node \a i1 of type 1
        const Neighbor & nb1( size_t i1, size_t _i2 ) const {
            DAI_DEBASSERT( i1 < _nb1.size() );
            DAI_DEBASSERT( _i2 < _nb1[i1].size() );
            return _nb1[i1][_i2];
        }
        /// Returns reference to the \a _i2 'th neighbor of node \a i1 of type 1
        Neighbor & nb1( size_t i1, size_t _i2 ) {
            DAI_DEBASSERT( i1 < _nb1.size() );
            DAI_DEBASSERT( _i2 < _nb1[i1].size() );
            return _nb1[i1][_i2];
        }

        /// Returns constant reference to the \a _i1 'th neighbor of node \a i2 of type 2
        const Neighbor & nb2( size_t i2, size_t _i1 ) const {
            DAI_DEBASSERT( i2 < _nb2.size() );
            DAI_DEBASSERT( _i1 < _nb2[i2].size() );
            return _nb2[i2][_i1];
        }
        /// Returns reference to the \a _i1 'th neighbor of node \a i2 of type 2
        Neighbor & nb2( size_t i2, size_t _i1 ) {
            DAI_DEBASSERT( i2 < _nb2.size() );
            DAI_DEBASSERT( _i1 < _nb2[i2].size() );
            return _nb2[i2][_i1];
        }

        /// Returns constant reference to all neighbors of node \a i1 of type 1
        const Neighbors & nb1( size_t i1 ) const {
            DAI_DEBASSERT( i1 < _nb1.size() );
            return _nb1[i1];
        }
        /// Returns reference to all neighbors of node \a i1 of type 1
        Neighbors & nb1( size_t i1 ) {
            DAI_DEBASSERT( i1 < _nb1.size() );
            return _nb1[i1];
        }

        /// Returns constant reference to all neighbors of node \a i2 of type 2
        const Neighbors & nb2( size_t i2 ) const {
            DAI_DEBASSERT( i2 < _nb2.size() );
            return _nb2[i2];
        }
        /// Returns reference to all neighbors of node \a i2 of type 2
        Neighbors & nb2( size_t i2 ) {
            DAI_DEBASSERT( i2 < _nb2.size() );
            return _nb2[i2];
        }
    //@}

    /// \name Adding nodes and edges
    //@{
        /// (Re)constructs BipartiteGraph from a range of edges.
        /** \tparam EdgeInputIterator Iterator that iterates over instances of Edge.
         *  \param nrNodes1 The number of nodes of type 1.
         *  \param nrNodes2 The number of nodes of type 2.
         *  \param begin Points to the first edge.
         *  \param end Points just beyond the last edge.
         *  \param check Whether to only add an edge if it does not exist already.
         */
        template<typename EdgeInputIterator>
        void construct( size_t nrNodes1, size_t nrNodes2, EdgeInputIterator begin, EdgeInputIterator end, bool check=true );

        /// Adds a node of type 1 without neighbors and returns the index of the added node.
        size_t addNode1() { _nb1.push_back( Neighbors() ); return _nb1.size() - 1; }

        /// Adds a node of type 2 without neighbors and returns the index of the added node.
        size_t addNode2() { _nb2.push_back( Neighbors() ); return _nb2.size() - 1; }


        /// Adds a node of type 1, with neighbors specified by a range of nodes of type 2.
        /** \tparam NodeInputIterator Iterator that iterates over instances of \c size_t.
         *  \param begin Points to the first index of the nodes of type 2 that should become neighbors of the added node.
         *  \param end Points just beyond the last index of the nodes of type 2 that should become neighbors of the added node.
         *  \param sizeHint For improved efficiency, the size of the range may be specified by \a sizeHint.
         *  \returns Index of the added node.
         */
        template <typename NodeInputIterator>
        size_t addNode1( NodeInputIterator begin, NodeInputIterator end, size_t sizeHint = 0 ) {
            Neighbors nbs1new;
            nbs1new.reserve( sizeHint );
            size_t iter = 0;
            for( NodeInputIterator it = begin; it != end; ++it ) {
                DAI_ASSERT( *it < nrNodes2() );
                Neighbor nb1new( iter, *it, nb2(*it).size() );
                Neighbor nb2new( nb2(*it).size(), nrNodes1(), iter++ );
                nbs1new.push_back( nb1new );
                nb2( *it ).push_back( nb2new );
            }
            _nb1.push_back( nbs1new );
            return _nb1.size() - 1;
        }

        /// Adds a node of type 2, with neighbors specified by a range of nodes of type 1.
        /** \tparam NodeInputIterator Iterator that iterates over instances of \c size_t.
         *  \param begin Points to the first index of the nodes of type 1 that should become neighbors of the added node.
         *  \param end Points just beyond the last index of the nodes of type 1 that should become neighbors of the added node.
         *  \param sizeHint For improved efficiency, the size of the range may be specified by \a sizeHint.
         *  \returns Index of the added node.
         */
        template <typename NodeInputIterator>
        size_t addNode2( NodeInputIterator begin, NodeInputIterator end, size_t sizeHint = 0 ) {
            Neighbors nbs2new;
            nbs2new.reserve( sizeHint );
            size_t iter = 0;
            for( NodeInputIterator it = begin; it != end; ++it ) {
                DAI_ASSERT( *it < nrNodes1() );
                Neighbor nb2new( iter, *it, nb1(*it).size() );
                Neighbor nb1new( nb1(*it).size(), nrNodes2(), iter++ );
                nbs2new.push_back( nb2new );
                nb1( *it ).push_back( nb1new );
            }
            _nb2.push_back( nbs2new );
            return _nb2.size() - 1;
        }

        /// Adds an edge between node \a n1 of type 1 and node \a n2 of type 2.
        /** If \a check == \c true, only adds the edge if it does not exist already.
         */
        BipartiteGraph& addEdge( size_t n1, size_t n2, bool check = true );
    //@}

    /// \name Erasing nodes and edges
    //@{
        /// Removes node \a n1 of type 1 and all incident edges; indices of other nodes are changed accordingly.
        void eraseNode1( size_t n1 );

        /// Removes node \a n2 of type 2 and all incident edges; indices of other nodes are changed accordingly.
        void eraseNode2( size_t n2 );

        /// Removes edge between node \a n1 of type 1 and node \a n2 of type 2.
        void eraseEdge( size_t n1, size_t n2 );
    //@}

    /// \name Queries
    //@{
        /// Returns number of nodes of type 1
        size_t nrNodes1() const { return _nb1.size(); }
        /// Returns number of nodes of type 2
        size_t nrNodes2() const { return _nb2.size(); }

        /// Calculates the number of edges, time complexity: O(nrNodes1())
        size_t nrEdges() const {
            size_t sum = 0;
            for( size_t i1 = 0; i1 < nrNodes1(); i1++ )
                sum += nb1(i1).size();
            return sum;
        }

        /// Returns true if the graph contains an edge between node \a n1 of type 1 and node \a n2 of type 2.
        /** \note The time complexity is linear in the number of neighbors of \a n1 or \a n2
         */
        bool hasEdge( size_t n1, size_t n2 ) const {
            if( nb1(n1).size() < nb2(n2).size() ) {
                for( size_t _n2 = 0; _n2 < nb1(n1).size(); _n2++ )
                    if( nb1( n1, _n2 ) == n2 )
                        return true;
            } else {
                for( size_t _n1 = 0; _n1 < nb2(n2).size(); _n1++ )
                    if( nb2( n2, _n1 ) == n1 )
                        return true;
            }
            return false;
        }

        /// Returns the index of a given node \a n2 of type 2 amongst the neighbors of node \a n1 of type 1
        /** \note The time complexity is linear in the number of neighbors of \a n1
         *  \throw OBJECT_NOT_FOUND if \a n2 is not a neighbor of \a n1
         */
        size_t findNb1( size_t n1, size_t n2 ) {
            for( size_t _n2 = 0; _n2 < nb1(n1).size(); _n2++ )
                if( nb1( n1, _n2 ) == n2 )
                    return _n2;
            DAI_THROW(OBJECT_NOT_FOUND);
            return nb1(n1).size();
        }

        /// Returns the index of a given node \a n1 of type 1 amongst the neighbors of node \a n2 of type 2
        /** \note The time complexity is linear in the number of neighbors of \a n2
         *  \throw OBJECT_NOT_FOUND if \a n1 is not a neighbor of \a n2
         */
        size_t findNb2( size_t n1, size_t n2 ) {
            for( size_t _n1 = 0; _n1 < nb2(n2).size(); _n1++ )
                if( nb2( n2, _n1 ) == n1 )
                    return _n1;
            DAI_THROW(OBJECT_NOT_FOUND);
            return nb2(n2).size();
        }

        /// Returns neighbors of node \a n1 of type 1 as a SmallSet<size_t>.
        SmallSet<size_t> nb1Set( size_t n1 ) const;

        /// Returns neighbors of node \a n2 of type 2 as a SmallSet<size_t>.
        SmallSet<size_t> nb2Set( size_t n2 ) const;

        /// Calculates second-order neighbors (i.e., neighbors of neighbors) of node \a n1 of type 1.
        /** If \a include == \c true, includes \a n1 itself, otherwise excludes \a n1.
         *  \note In libDAI versions 0.2.4 and earlier, this function used to return a std::vector<size_t>
         */
        SmallSet<size_t> delta1( size_t n1, bool include = false ) const;

        /// Calculates second-order neighbors (i.e., neighbors of neighbors) of node \a n2 of type 2.
        /** If \a include == \c true, includes \a n2 itself, otherwise excludes \a n2.
         *  \note In libDAI versions 0.2.4 and earlier, this function used to return a std::vector<size_t>
         */
        SmallSet<size_t> delta2( size_t n2, bool include = false ) const;

        /// Returns true if the graph is connected
        bool isConnected() const;

        /// Returns true if the graph is a tree, i.e., if it is singly connected and connected.
        bool isTree() const;

        /// Comparison operator which returns true if two graphs are identical
        /** \note Two graphs are called identical if they have the same number of nodes
         *  of both types and the same edges (i.e., \a x has an edge between nodes
         *  \a n1 and \a n2 if and only if \c *this has an edge between nodes \a n1 and \a n2).
         */
        bool operator==( const BipartiteGraph& x ) const {
            if( nrNodes1() != x.nrNodes1() )
                return false;
            if( nrNodes2() != x.nrNodes2() )
                return false;
            for( size_t n1 = 0; n1 < nrNodes1(); n1++ ) {
                if( nb1(n1).size() != x.nb1(n1).size() )
                    return false;
                foreach( const Neighbor &n2, nb1(n1) )
                    if( !x.hasEdge( n1, n2 ) )
                        return false;
                foreach( const Neighbor &n2, x.nb1(n1) )
                    if( !hasEdge( n1, n2 ) )
                        return false;
            }
            return true;
        }

        /// Asserts internal consistency
        void checkConsistency() const;
    //@}

    /// \name Input and output
    //@{
        /// Writes this BipartiteGraph to an output stream in GraphViz .dot syntax
        void printDot( std::ostream& os ) const;

        /// Writes this BipartiteGraph to an output stream
        friend std::ostream& operator<<( std::ostream& os, const BipartiteGraph& g ) {
            g.printDot( os );
            return os;
        }
    //@}
};


template<typename EdgeInputIterator>
void BipartiteGraph::construct( size_t nrNodes1, size_t nrNodes2, EdgeInputIterator begin, EdgeInputIterator end, bool check ) {
    _nb1.clear();
    _nb1.resize( nrNodes1 );
    _nb2.clear();
    _nb2.resize( nrNodes2 );

    for( EdgeInputIterator e = begin; e != end; e++ )
        addEdge( e->first, e->second, check );
}


} // end of namespace dai


/** \example example_bipgraph.cpp
 *  This example deals with the following bipartite graph:
 *  \dot
 *  graph example {
 *    ordering=out;
 *    subgraph cluster_type1 {
 *      node[shape=circle,width=0.4,fixedsize=true,style=filled];
 *      12 [label="2"];
 *      11 [label="1"];
 *      10 [label="0"];
 *    }
 *    subgraph cluster_type2 {
 *      node[shape=polygon,regular=true,sides=4,width=0.4,fixedsize=true,style=filled];
 *      21 [label="1"];
 *      20 [label="0"];
 *    }
 *    10 -- 20;
 *    11 -- 20;
 *    12 -- 20;
 *    11 -- 21;
 *    12 -- 21;
 *  }
 *  \enddot
 *  It has three nodes of type 1 (drawn as circles) and two nodes of type 2 (drawn as rectangles).
 *  Node 0 of type 1 has only one neighbor (node 0 of type 2), but node 0 of type 2 has three neighbors (nodes 0,1,2 of type 1).
 *  The example code shows how to construct a BipartiteGraph object representing this bipartite graph and
 *  how to iterate over nodes and their neighbors.
 *
 *  \section Output
 *  \verbinclude examples/example_bipgraph.out
 *
 *  \section Source
 */


#endif
