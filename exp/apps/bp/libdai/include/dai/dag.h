/*  This file is part of libDAI - http://www.libdai.org/
 *
 *  Copyright (c) 2006-2011, The libDAI authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
 */


/// \file
/// \brief Defines the DAG class, which represents a directed acyclic graph


#ifndef __defined_libdai_dag_h
#define __defined_libdai_dag_h


#include <ostream>
#include <vector>
#include <algorithm>
#include <dai/util.h>
#include <dai/exceptions.h>
#include <dai/smallset.h>
#include <dai/graph.h>


namespace dai {


/// Represents the neighborhood structure of nodes in a directed cyclic graph.
/** A directed cyclic graph has nodes connected by directed edges, such that there is no
 *  directed cycle of edges n1->n2->n3->...->n1. Nodes are indexed by an unsigned integer. 
 *  If there are nrNodes() nodes, they are numbered 0,1,2,...,nrNodes()-1. An edge
 *  from node \a n1 to node \a n2 is represented by a Edge(\a n1,\a n2).
 *
 *  DAG is implemented as a sparse adjacency list, i.e., it stores for each node a list of
 *  its parents and a list of its children. Both lists are implemented as a vector of Neighbor
 *  structures (accessible by the pa() and ch() methods). Thus, each node has two associated 
 *  variables of type DAG::Neighbors, which are vectors of Neighbor structures, describing their
 *  parent and children nodes.
 */
class DAG {
    private:
        /// Contains for each node a vector of its parent nodes
        std::vector<Neighbors> _pa;
        
        /// Contains for each node a vector of its children nodes
        std::vector<Neighbors> _ch;

    public:
    /// \name Constructors and destructors
    //@{
        /// Default constructor (creates an empty DAG).
        DAG() : _pa(), _ch() {}

        /// Constructs DAG with \a nr nodes and no edges.
        DAG( size_t nr ) : _pa( nr ), _ch( nr ) {}

        /// Constructs DAG from a range of edges.
        /** \tparam EdgeInputIterator Iterator that iterates over instances of Edge.
         *  \param nr The number of nodes.
         *  \param begin Points to the first edge.
         *  \param end Points just beyond the last edge.
         *  \param check Whether to only add an edge if it does not exist already and
         *    if it does not introduce a cycle
         */
        template<typename EdgeInputIterator>
        DAG( size_t nr, EdgeInputIterator begin, EdgeInputIterator end, bool check=true ) : _pa(), _ch() {
            construct( nr, begin, end, check );
        }
    //@}

    /// \name Accessors and mutators
    //@{
        /// Returns constant reference to the \a _p 'th parent of node \a n
        const Neighbor& pa( size_t n, size_t _p ) const {
            DAI_DEBASSERT( n < _pa.size() );
            DAI_DEBASSERT( _p < _pa[n].size() );
            return _pa[n][_p];
        }
        /// Returns reference to the \a _p 'th parent of node \a n
        Neighbor& pa( size_t n, size_t _p ) {
            DAI_DEBASSERT( n < _pa.size() );
            DAI_DEBASSERT( _p < _pa[n].size() );
            return _pa[n][_p];
        }

        /// Returns constant reference to all parents of node \a n
        const Neighbors& pa( size_t n ) const {
            DAI_DEBASSERT( n < _pa.size() );
            return _pa[n];
        }
        /// Returns reference to all parents of node \a n
        Neighbors& pa( size_t n ) {
            DAI_DEBASSERT( n < _pa.size() );
            return _pa[n];
        }

        /// Returns constant reference to the \a _c 'th child of node \a n
        const Neighbor& ch( size_t n, size_t _c ) const {
            DAI_DEBASSERT( n < _ch.size() );
            DAI_DEBASSERT( _c < _ch[n].size() );
            return _ch[n][_c];
        }
        /// Returns reference to the \a _c 'th child of node \a n
        Neighbor& ch( size_t n, size_t _c ) {
            DAI_DEBASSERT( n < _ch.size() );
            DAI_DEBASSERT( _c < _ch[n].size() );
            return _ch[n][_c];
        }

        /// Returns constant reference to all children of node \a n
        const Neighbors& ch( size_t n ) const {
            DAI_DEBASSERT( n < _ch.size() );
            return _ch[n];
        }
        /// Returns reference to all children of node \a n
        Neighbors& ch( size_t n ) {
            DAI_DEBASSERT( n < _ch.size() );
            return _ch[n];
        }
    //@}

    /// \name Adding nodes and edges
    //@{
        /// (Re)constructs DAG from a range of edges.
        /** \tparam EdgeInputIterator Iterator that iterates over instances of Edge.
         *  \param nr The number of nodes.
         *  \param begin Points to the first edge.
         *  \param end Points just beyond the last edge.
         *  \param check Whether to only add an edge if it does not exist already and does not introduce a cycle.
         */
        template<typename EdgeInputIterator>
        void construct( size_t nr, EdgeInputIterator begin, EdgeInputIterator end, bool check=true );

        /// Adds a node without parents and children and returns the index of the added node.
        size_t addNode() {
            _pa.push_back( Neighbors() ); 
            _ch.push_back( Neighbors() );
            return _pa.size() - 1;
        }

        /// Adds a node with parents specified by a range of nodes.
        /** \tparam NodeInputIterator Iterator that iterates over instances of \c size_t.
         *  \param begin Points to the first index of the nodes that should become parents of the added node.
         *  \param end Points just beyond the last index of the nodes that should become parents of the added node.
         *  \param sizeHint For improved efficiency, the size of the range may be specified by \a sizeHint.
         *  \returns Index of the added node.
         */
        template <typename NodeInputIterator>
        size_t addNode( NodeInputIterator begin, NodeInputIterator end, size_t sizeHint=0 ) {
            Neighbors newparents;
            newparents.reserve( sizeHint );
            size_t iter = 0;
            for( NodeInputIterator it = begin; it != end; ++it ) {
                DAI_ASSERT( *it < nrNodes() );
                Neighbor newparent( iter, *it, ch(*it).size() );
                Neighbor newchild( ch(*it).size(), nrNodes(), iter++ );
                newparents.push_back( newparent );
                ch( *it ).push_back( newchild );
            }
            _pa.push_back( newparents );
            _ch.push_back( Neighbors() );
            return _pa.size() - 1;
        }

        /// Adds an edge from node \a n1 and node \a n2.
        /** If \a check == \c true, only adds the edge if it does not exist already and it would not introduce a cycle.
         */
        DAG& addEdge( size_t n1, size_t n2, bool check=true );
    //@}

    /// \name Erasing nodes and edges
    //@{
        /// Removes node \a n and all ingoing and outgoing edges; indices of other nodes are changed accordingly.
        void eraseNode( size_t n );

        /// Removes edge from node \a n1 to node \a n2.
        void eraseEdge( size_t n1, size_t n2 );
    //@}

    /// \name Queries
    //@{
        /// Returns number of nodes
        size_t nrNodes() const {
            DAI_DEBASSERT( _pa.size() == _ch.size() );
            return _pa.size();
        }

        /// Calculates the number of edges, time complexity: O(nrNodes())
        size_t nrEdges() const {
            size_t sum = 0;
            for( size_t i = 0; i < _pa.size(); i++ )
                sum += _pa[i].size();
            return sum;
        }

        /// Returns true if the DAG contains an edge from node \a n1 and \a n2
        /** \note The time complexity is linear in the number of children of \a n1 or the number of parents of \a n2, whichever is smaller
         */
        bool hasEdge( size_t n1, size_t n2 ) const {
            if( ch(n1).size() < pa(n2).size() ) {
                for( size_t _n2 = 0; _n2 < ch(n1).size(); _n2++ )
                    if( ch( n1, _n2 ) == n2 )
                        return true;
            } else {
                for( size_t _n1 = 0; _n1 < pa(n2).size(); _n1++ )
                    if( pa( n2, _n1 ) == n1 )
                        return true;
            }
            return false;
        }

        /// Returns the index of a given node \a p amongst the parents of \a n
        /** \note The time complexity is linear in the number of parents of \a n
         *  \throw OBJECT_NOT_FOUND if \a p is not a parent of \a n
         */
        size_t findPa( size_t n, size_t p ) {
            for( size_t _p = 0; _p < pa(n).size(); _p++ )
                if( pa( n, _p ) == p )
                    return _p;
            DAI_THROW(OBJECT_NOT_FOUND);
            return pa(n).size();
        }

        /// Returns the index of a given node \a c amongst the children of \a n
        /** \note The time complexity is linear in the number of children of \a n
         *  \throw OBJECT_NOT_FOUND if \a c is not a child \a n
         */
        size_t findCh( size_t n, size_t c ) {
            for( size_t _c = 0; _c < ch(n).size(); _c++ )
                if( ch( n, _c ) == c )
                    return _c;
            DAI_THROW(OBJECT_NOT_FOUND);
            return ch(n).size();
        }

        /// Returns parents of node \a n as a SmallSet<size_t>.
        SmallSet<size_t> paSet( size_t n ) const;

        /// Returns children of node \a n as a SmallSet<size_t>.
        SmallSet<size_t> chSet( size_t n ) const;

        /// Returns the set of ancestors of node \a n, i.e., all nodes \a a such that there exists a directed path from \a a to \a n (excluding \a n itself)
        std::set<size_t> ancestors( size_t n ) const;

        /// Returns the set of descendants of node \a n, i.e., all nodes \a d such that there exists a directed path from \a n to \a d (excluding \a n itself)
        std::set<size_t> descendants( size_t n ) const;

        /// Returns whether there exists a directed path from node \a n1 to node \a n2 (which may consist of zero edges)
        bool existsDirectedPath( size_t n1, size_t n2 ) const;

        /// Returns true if the DAG is connected
        bool isConnected() const;

        /// Asserts internal consistency
        void checkConsistency() const;

        /// Comparison operator which returns true if two DAGs are identical
        /** \note Two DAGs are called identical if they have the same number 
         *  of nodes and the same edges (i.e., \a x has an edge from \a n1 to \a n2
         *  if and only if \c *this has an edge from node \a n1 to \a n2).
         */
        bool operator==( const DAG& x ) const {
            if( nrNodes() != x.nrNodes() )
                return false;
            for( size_t n1 = 0; n1 < nrNodes(); n1++ ) {
                if( pa(n1).size() != x.pa(n1).size() )
                    return false;
                foreach( const Neighbor &n2, pa(n1) )
                    if( !x.hasEdge( n2, n1 ) )
                        return false;
                foreach( const Neighbor &n2, x.pa(n1) )
                    if( !hasEdge( n2, n1 ) )
                        return false;
            }
            return true;
        }
    //@}

    /// \name Input and output
    //@{
        /// Writes this DAG to an output stream in GraphViz .dot syntax
        void printDot( std::ostream& os ) const;

        /// Writes this DAG to an output stream
        friend std::ostream& operator<<( std::ostream& os, const DAG& g ) {
            g.printDot( os );
            return os;
        }
    //@}
};


template<typename EdgeInputIterator>
void DAG::construct( size_t nr, EdgeInputIterator begin, EdgeInputIterator end, bool check ) {
    _pa.clear();
    _pa.resize( nr );
    _ch.clear();
    _ch.resize( nr );

    for( EdgeInputIterator e = begin; e != end; e++ )
        addEdge( e->first, e->second, check );
}


} // end of namespace dai


#endif
