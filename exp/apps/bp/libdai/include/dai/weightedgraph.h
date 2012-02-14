/*  This file is part of libDAI - http://www.libdai.org/
 *
 *  Copyright (c) 2006-2011, The libDAI authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
 */


/** \file
 *  \brief Defines some utility functions for (weighted) undirected graphs, trees and rooted trees.
 *  \todo Improve general support for graphs and trees (in particular, a good tree implementation is needed).
 */


#ifndef __defined_libdai_weightedgraph_h
#define __defined_libdai_weightedgraph_h


#include <vector>
#include <map>
#include <iostream>
#include <set>
#include <limits>
#include <climits>   // Work-around for bug in boost graph library
#include <dai/util.h>
#include <dai/exceptions.h>
#include <dai/graph.h>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/prim_minimum_spanning_tree.hpp>
#include <boost/graph/kruskal_min_spanning_tree.hpp>


namespace dai {


/// Represents a directed edge
class DEdge {
    public:
        /// First node index (source of edge)
        size_t first;

        /// Second node index (target of edge)
        size_t second;

        /// Default constructor
        DEdge() : first(0), second(0) {}

        /// Constructs a directed edge pointing from \a m1 to \a m2
        DEdge( size_t m1, size_t m2 ) : first(m1), second(m2) {}

        /// Tests for equality
        bool operator==( const DEdge &x ) const { return ((first == x.first) && (second == x.second)); }

        /// Smaller-than operator (performs lexicographical comparison)
        bool operator<( const DEdge &x ) const {
            return( (first < x.first) || ((first == x.first) && (second < x.second)) );
        }

        /// Writes a directed edge to an output stream
        friend std::ostream & operator << (std::ostream & os, const DEdge & e) {
            os << "(" << e.first << "->" << e.second << ")";
            return os;
        }
};


/// Represents an undirected edge
class UEdge {
    public:
        /// First node index
        size_t first;

        /// Second node index
        size_t second;

        /// Default constructor
        UEdge() : first(0), second(0) {}

        /// Constructs an undirected edge between \a m1 and \a m2
        UEdge( size_t m1, size_t m2 ) : first(m1), second(m2) {}

        /// Construct from DEdge
        UEdge( const DEdge &e ) : first(e.first), second(e.second) {}

        /// Tests for inequality (disregarding the ordering of the nodes)
        bool operator==( const UEdge &x ) {
            return ((first == x.first) && (second == x.second)) || ((first == x.second) && (second == x.first));
        }

        /// Smaller-than operator
        bool operator<( const UEdge &x ) const {
            size_t s = std::min( first, second );
            size_t l = std::max( first, second );
            size_t xs = std::min( x.first, x.second );
            size_t xl = std::max( x.first, x.second );
            return( (s < xs) || ((s == xs) && (l < xl)) );
        }

        /// Writes an undirected edge to an output stream
        friend std::ostream & operator << (std::ostream & os, const UEdge & e) {
            if( e.first < e.second )
                os << "{" << e.first << "--" << e.second << "}";
            else
                os << "{" << e.second << "--" << e.first << "}";
            return os;
        }
};


/// Represents an undirected graph, implemented as a std::set of undirected edges
class GraphEL : public std::set<UEdge> {
    public:
        /// Default constructor
        GraphEL() {}

        /// Construct from range of objects that can be cast to UEdge
        template <class InputIterator>
        GraphEL( InputIterator begin, InputIterator end ) {
            insert( begin, end );
        }

        /// Construct from GraphAL
        GraphEL( const GraphAL& G ) {
            for( size_t n1 = 0; n1 < G.nrNodes(); n1++ )
                foreach( const Neighbor n2, G.nb(n1) )
                    if( n1 < n2 )
                        insert( UEdge( n1, n2 ) );
        }
};


/// Represents an undirected weighted graph, with weights of type \a T, implemented as a std::map mapping undirected edges to weights
template<class T> class WeightedGraph : public std::map<UEdge, T> {};


/// Represents a rooted tree, implemented as a vector of directed edges
/** By convention, the edges are stored such that they point away from 
 *  the root and such that edges nearer to the root come before edges
 *  farther away from the root.
 */
class RootedTree : public std::vector<DEdge> {
    public:
        /// Default constructor
        RootedTree() {}

        /// Constructs a rooted tree from a tree and a root
        /** \pre T has no cycles and contains node \a Root
         */
        RootedTree( const GraphEL &T, size_t Root );
};


/// Constructs a minimum spanning tree from the (non-negatively) weighted graph \a G.
/** \param G Weighted graph that should have non-negative weights.
 *  \param usePrim If true, use Prim's algorithm (complexity O(E log(V))), otherwise, use Kruskal's algorithm (complexity O(E log(E))).
 *  \note Uses implementation from Boost Graph Library.
 *  \note The vertices of \a G must be in the range [0,N) where N is the number of vertices of \a G.
 */
template<typename T> RootedTree MinSpanningTree( const WeightedGraph<T> &G, bool usePrim ) {
    RootedTree result;
    if( G.size() > 0 ) {
        using namespace boost;
        using namespace std;
        typedef adjacency_list< vecS, vecS, undirectedS, no_property, property<edge_weight_t, double> > boostGraph;

        set<size_t> nodes;
        vector<UEdge> edges;
        vector<double> weights;
        edges.reserve( G.size() );
        weights.reserve( G.size() );
        for( typename WeightedGraph<T>::const_iterator e = G.begin(); e != G.end(); e++ ) {
            weights.push_back( e->second );
            edges.push_back( e->first );
            nodes.insert( e->first.first );
            nodes.insert( e->first.second );
        }

        size_t N = nodes.size();
        for( set<size_t>::const_iterator it = nodes.begin(); it != nodes.end(); it++ )
            if( *it >= N )
                DAI_THROWE(RUNTIME_ERROR,"Vertices must be in range [0..N) where N is the number of vertices.");

        boostGraph g( edges.begin(), edges.end(), weights.begin(), nodes.size() );
        size_t root = *(nodes.begin());
        GraphEL tree;
        if( usePrim ) {
            // Prim's algorithm
            vector< graph_traits< boostGraph >::vertex_descriptor > p(N);
            prim_minimum_spanning_tree( g, &(p[0]) );

            // Store tree edges in result
            for( size_t i = 0; i != p.size(); i++ ) {
                if( p[i] != i )
                    tree.insert( UEdge( p[i], i ) );
            }
        } else {
            // Kruskal's algorithm
            vector< graph_traits< boostGraph >::edge_descriptor > t;
            t.reserve(  N - 1 );
            kruskal_minimum_spanning_tree( g, std::back_inserter(t) );

            // Store tree edges in result
            for( size_t i = 0; i != t.size(); i++ ) {
                size_t v1 = source( t[i], g );
                size_t v2 = target( t[i], g );
                if( v1 != v2 )
                    tree.insert( UEdge( v1, v2 ) );
            }
        }

        // Direct edges in order to obtain a rooted tree
        result = RootedTree( tree, root );
    }
    return result;
}


/// Constructs a minimum spanning tree from the (non-negatively) weighted graph \a G.
/** \param G Weighted graph that should have non-negative weights.
 *  \param usePrim If true, use Prim's algorithm (complexity O(E log(V))), otherwise, use Kruskal's algorithm (complexity O(E log(E))).
 *  \note Uses implementation from Boost Graph Library.
 *  \note The vertices of \a G must be in the range [0,N) where N is the number of vertices of \a G.
 */
template<typename T> RootedTree MaxSpanningTree( const WeightedGraph<T> &G, bool usePrim ) {
    if( G.size() == 0 )
        return RootedTree();
    else {
        T maxweight = G.begin()->second;
        for( typename WeightedGraph<T>::const_iterator it = G.begin(); it != G.end(); it++ )
            if( it->second > maxweight )
                maxweight = it->second;
        // make a copy of the graph
        WeightedGraph<T> gr( G );
        // invoke MinSpanningTree with negative weights
        // (which have to be shifted to satisfy positivity criterion)
        for( typename WeightedGraph<T>::iterator it = gr.begin(); it != gr.end(); it++ )
            it->second = maxweight - it->second;
        return MinSpanningTree( gr, usePrim );
    }
}


} // end of namespace dai


#endif
