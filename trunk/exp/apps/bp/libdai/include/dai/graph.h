/*  This file is part of libDAI - http://www.libdai.org/
 *
 *  Copyright (c) 2006-2011, The libDAI authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
 */


/// \file
/// \brief Defines the GraphAL class, which represents an undirected graph as an adjacency list


#ifndef __defined_libdai_graph_h
#define __defined_libdai_graph_h


#include <ostream>
#include <vector>
#include <algorithm>
#include <dai/util.h>
#include <dai/exceptions.h>
#include <dai/smallset.h>


namespace dai {


/// Describes the neighbor relationship of two nodes in a graph.
/** Most graphs that libDAI deals with are sparse. Therefore,
 *  a fast and memory-efficient way of representing the structure
 *  of a sparse graph is needed. A frequently used operation that
 *  also needs to be fast is switching between viewing node \a a as a 
 *  neighbor of node \a b, and node \a b as a neighbor of node \a a.
 *  The Neighbor struct solves both of these problems.
 *
 *  Most sparse graphs in libDAI are represented by storing for each
 *  node in the graph the set of its neighbors. In practice, this set
 *  of neighbors is stored using the Neighbors type, which is simply a
 *  std::vector<\link Neighbor \endlink>. The Neighbor struct contains
 *  the label of the neighboring node (the \c node member) and 
 *  additional information which allows to access a node as a neighbor 
 *  of its neighbor (the \c dual member). For convenience, each Neighbor 
 *  structure also stores its index in the Neighbors vector that it is 
 *  part of (the \c iter member).
 *
 *  By convention, variable identifiers naming indices into a vector
 *  of neighbors are prefixed with an underscore ("_"). The neighbor list 
 *  which they point into is then understood from the context.
 *
 *  Let us denote the \a _j 'th neighbor of node \a i by <tt>nb(i,_j)</tt>,
 *  which is of the Neighbor type. Here, \a i is the "absolute" index of 
 *  node \a i, but \a _j is understood as a "relative" index, giving node 
 *  \a j 's entry in the Neighbors <tt>nb(i)</tt> of node \a i. The absolute
 *  index of \a _j, which would be denoted \a j, can be recovered from the 
 *  \c node member, <tt>nb(i,_j).node</tt>. The \c iter member 
 *  <tt>nb(i,_j).iter</tt> gives the relative index \a _j, and the \c dual 
 *  member <tt>nb(i,_j).dual</tt> gives the "dual" relative index, i.e., 
 *  the index of \a i in \a j 's neighbor list.
 *
 *  Iteration over edges can be easily accomplished:
 *  \code
 *  for( size_t i = 0; i < nrNodes(); ++i ) {
 *      size_t _j = 0;
 *      foreach( const Neighbor &j, nb(i) ) {
 *          assert( j == nb(i,j.iter) );
 *          assert( nb(j.node,j.dual).node == i );
 *          assert( _j = j.iter );
 *          _j++;
 *      }
 *  }
 *  \endcode
 */
struct Neighbor {
    /// Corresponds to the index of this Neighbor entry in the vector of neighbors
    size_t iter;
    /// Contains the absolute index of the neighboring node
    size_t node;
    /// Contains the "dual" index (i.e., the index of this node in the Neighbors vector of the neighboring node)
    size_t dual;

    /// Default constructor
    Neighbor() {}
    /// Constructor that allows setting the values of the member variables
    Neighbor( size_t iter, size_t node, size_t dual ) : iter(iter), node(node), dual(dual) {}

    /// Cast to \c size_t returns \c node member
    operator size_t () const { return node; }
};


/// Describes the set of neighbors of some node in a graph
typedef std::vector<Neighbor> Neighbors;


/// Represents an edge in a graph: an Edge(\a i,\a j) corresponds to the edge between node \a i and node \a j.
/** \note If the edge is interpreted as a directed edge, then it points from \a i to \a j.
 *  \note If the edge is part of a bipartite graph, \a i is understood to correspond to a node of type 1, and
 *  \a j to a node of type 2.
 */
typedef std::pair<size_t,size_t> Edge;


/// Represents the neighborhood structure of nodes in an undirected graph.
/** A graph has nodes connected by edges. Nodes are indexed by an unsigned integer. 
 *  If there are nrNodes() nodes, they are numbered 0,1,2,...,nrNodes()-1. An edge
 *  between node \a n1 and node \a n2 is represented by a Edge(\a n1,\a n2).
 *
 *  GraphAL is implemented as a sparse adjacency list, i.e., it stores for each node a list of
 *  its neighboring nodes. The list of neighboring nodes is implemented as a vector of Neighbor
 *  structures (accessible by the nb() method). Thus, each node has an associated variable of 
 *  type GraphAL::Neighbors, which is a vector of Neighbor structures, describing its 
 *  neighboring nodes.
 */
class GraphAL {
    private:
        /// Contains for each node a vector of its neighbors
        std::vector<Neighbors> _nb;

    public:
    /// \name Constructors and destructors
    //@{
        /// Default constructor (creates an empty graph).
        GraphAL() : _nb() {}

        /// Constructs GraphAL with \a nr nodes and no edges.
        GraphAL( size_t nr ) : _nb( nr ) {}

        /// Constructs GraphAL from a range of edges.
        /** \tparam EdgeInputIterator Iterator that iterates over instances of Edge.
         *  \param nr The number of nodes.
         *  \param begin Points to the first edge.
         *  \param end Points just beyond the last edge.
         *  \param check Whether to only add an edge if it does not exist already.
         */
        template<typename EdgeInputIterator>
        GraphAL( size_t nr, EdgeInputIterator begin, EdgeInputIterator end, bool check=true ) : _nb() {
            construct( nr, begin, end, check );
        }
    //@}

    /// \name Accessors and mutators
    //@{
        /// Returns constant reference to the \a _n2 'th neighbor of node \a n1
        const Neighbor & nb( size_t n1, size_t _n2 ) const {
            DAI_DEBASSERT( n1 < _nb.size() );
            DAI_DEBASSERT( _n2 < _nb[n1].size() );
            return _nb[n1][_n2];
        }
        /// Returns reference to the \a _n2 'th neighbor of node \a n1
        Neighbor & nb( size_t n1, size_t _n2 ) {
            DAI_DEBASSERT( n1 < _nb.size() );
            DAI_DEBASSERT( _n2 < _nb[n1].size() );
            return _nb[n1][_n2];
        }

        /// Returns constant reference to all neighbors of node \a n
        const Neighbors & nb( size_t n ) const {
            DAI_DEBASSERT( n < _nb.size() );
            return _nb[n];
        }
        /// Returns reference to all neighbors of node \a n
        Neighbors & nb( size_t n ) {
            DAI_DEBASSERT( n < _nb.size() );
            return _nb[n];
        }
    //@}

    /// \name Adding nodes and edges
    //@{
        /// (Re)constructs GraphAL from a range of edges.
        /** \tparam EdgeInputIterator Iterator that iterates over instances of Edge.
         *  \param nr The number of nodes.
         *  \param begin Points to the first edge.
         *  \param end Points just beyond the last edge.
         *  \param check Whether to only add an edge if it does not exist already.
         */
        template<typename EdgeInputIterator>
        void construct( size_t nr, EdgeInputIterator begin, EdgeInputIterator end, bool check=true );

        /// Adds a node without neighbors and returns the index of the added node.
        size_t addNode() { _nb.push_back( Neighbors() ); return _nb.size() - 1; }

        /// Adds a node, with neighbors specified by a range of nodes.
        /** \tparam NodeInputIterator Iterator that iterates over instances of \c size_t.
         *  \param begin Points to the first index of the nodes that should become neighbors of the added node.
         *  \param end Points just beyond the last index of the nodes that should become neighbors of the added node.
         *  \param sizeHint For improved efficiency, the size of the range may be specified by \a sizeHint.
         *  \returns Index of the added node.
         */
        template <typename NodeInputIterator>
        size_t addNode( NodeInputIterator begin, NodeInputIterator end, size_t sizeHint = 0 ) {
            Neighbors nbsnew;
            nbsnew.reserve( sizeHint );
            size_t iter = 0;
            for( NodeInputIterator it = begin; it != end; ++it ) {
                DAI_ASSERT( *it < nrNodes() );
                Neighbor nb1new( iter, *it, nb(*it).size() );
                Neighbor nb2new( nb(*it).size(), nrNodes(), iter++ );
                nbsnew.push_back( nb1new );
                nb( *it ).push_back( nb2new );
            }
            _nb.push_back( nbsnew );
            return _nb.size() - 1;
        }

        /// Adds an edge between node \a n1 and node \a n2.
        /** If \a check == \c true, only adds the edge if it does not exist already.
         */
        GraphAL& addEdge( size_t n1, size_t n2, bool check = true );
    //@}

    /// \name Erasing nodes and edges
    //@{
        /// Removes node \a n and all incident edges; indices of other nodes are changed accordingly.
        void eraseNode( size_t n );

        /// Removes edge between node \a n1 and node \a n2.
        void eraseEdge( size_t n1, size_t n2 );
    //@}

    /// \name Queries
    //@{
        /// Returns number of nodes
        size_t nrNodes() const { return _nb.size(); }

        /// Calculates the number of edges, time complexity: O(nrNodes())
        size_t nrEdges() const {
            size_t sum = 0;
            for( size_t i = 0; i < nrNodes(); i++ )
                sum += nb(i).size();
            return sum / 2;
        }

        /// Returns true if the graph contains an edge between nodes \a n1 and \a n2
        /** \note The time complexity is linear in the number of neighbors of \a n1 or \a n2
         */
        bool hasEdge( size_t n1, size_t n2 ) const {
            if( nb(n1).size() < nb(n2).size() ) {
                for( size_t _n2 = 0; _n2 < nb(n1).size(); _n2++ )
                    if( nb( n1, _n2 ) == n2 )
                        return true;
            } else {
                for( size_t _n1 = 0; _n1 < nb(n2).size(); _n1++ )
                    if( nb( n2, _n1 ) == n1 )
                        return true;
            }
            return false;
        }

        /// Returns the index of a given node \a n2 amongst the neighbors of \a n1
        /** \note The time complexity is linear in the number of neighbors of \a n1
         *  \throw OBJECT_NOT_FOUND if \a n2 is not a neighbor of \a n1
         */
        size_t findNb( size_t n1, size_t n2 ) {
            for( size_t _n2 = 0; _n2 < nb(n1).size(); _n2++ )
                if( nb( n1, _n2 ) == n2 )
                    return _n2;
            DAI_THROW(OBJECT_NOT_FOUND);
            return nb(n1).size();
        }

        /// Returns neighbors of node \a n as a SmallSet<size_t>.
        SmallSet<size_t> nbSet( size_t n ) const;

        /// Returns true if the graph is connected
        bool isConnected() const;

        /// Returns true if the graph is a tree, i.e., if it is singly connected and connected.
        bool isTree() const;

        /// Asserts internal consistency
        void checkConsistency() const;

        /// Comparison operator which returns true if two graphs are identical
        /** \note Two graphs are called identical if they have the same number 
         *  of nodes and the same edges (i.e., \a x has an edge between nodes
         *  \a n1 and \a n2 if and only if \c *this has an edge between nodes \a n1 and \a n2).
         */
        bool operator==( const GraphAL& x ) const {
            if( nrNodes() != x.nrNodes() )
                return false;
            for( size_t n1 = 0; n1 < nrNodes(); n1++ ) {
                if( nb(n1).size() != x.nb(n1).size() )
                    return false;
                foreach( const Neighbor &n2, nb(n1) )
                    if( !x.hasEdge( n1, n2 ) )
                        return false;
                foreach( const Neighbor &n2, x.nb(n1) )
                    if( !hasEdge( n1, n2 ) )
                        return false;
            }
            return true;
        }
    //@}

    /// \name Input and output
    //@{
        /// Writes this GraphAL to an output stream in GraphViz .dot syntax
        void printDot( std::ostream& os ) const;

        /// Writes this GraphAL to an output stream
        friend std::ostream& operator<<( std::ostream& os, const GraphAL& g ) {
            g.printDot( os );
            return os;
        }
    //@}
};


template<typename EdgeInputIterator>
void GraphAL::construct( size_t nr, EdgeInputIterator begin, EdgeInputIterator end, bool check ) {
    _nb.clear();
    _nb.resize( nr );

    for( EdgeInputIterator e = begin; e != end; e++ )
        addEdge( e->first, e->second, check );
}


/// Creates a fully-connected graph with \a N nodes
GraphAL createGraphFull( size_t N );
/// Creates a two-dimensional rectangular grid of \a N1 by \a N2 nodes, which can be \a periodic
GraphAL createGraphGrid( size_t N1, size_t N2, bool periodic );
/// Creates a three-dimensional rectangular grid of \a N1 by \a N2 by \a N3 nodes, which can be \a periodic
GraphAL createGraphGrid3D( size_t N1, size_t N2, size_t N3, bool periodic );
/// Creates a graph consisting of a single loop of \a N nodes
GraphAL createGraphLoop( size_t N );
/// Creates a random tree-structured graph of \a N nodes
GraphAL createGraphTree( size_t N );
/// Creates a random regular graph of \a N nodes with uniform connectivity \a d
/** Algorithm 1 in [\ref StW99].
 *  Draws a random graph of size \a N and uniform degree \a d
 *  from an almost uniform probability distribution over these graphs
 *  (which becomes uniform in the limit that \a d is small and \a N goes
 *  to infinity).
 */
GraphAL createGraphRegular( size_t N, size_t d );


} // end of namespace dai


#endif
