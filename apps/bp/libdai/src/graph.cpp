/*  This file is part of libDAI - http://www.libdai.org/
 *
 *  Copyright (c) 2006-2011, The libDAI authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
 */


#include <dai/graph.h>


namespace dai {


using namespace std;


GraphAL& GraphAL::addEdge( size_t n1, size_t n2, bool check ) {
    DAI_ASSERT( n1 < nrNodes() );
    DAI_ASSERT( n2 < nrNodes() );
    bool exists = false;
    if( check ) {
        // Check whether the edge already exists
        foreach( const Neighbor &n, nb(n1) )
            if( n == n2 ) {
                exists = true;
                break;
            }
    }
    if( !exists && n1 != n2 ) { // Add edge
        Neighbor nb_1( nb(n1).size(), n2, nb(n2).size() );
        Neighbor nb_2( nb_1.dual, n1, nb_1.iter );
        nb(n1).push_back( nb_1 );
        nb(n2).push_back( nb_2 );
    }
    return *this;
}


void GraphAL::eraseNode( size_t n ) {
    DAI_ASSERT( n < nrNodes() );
    // Erase neighbor entry of node n
    _nb.erase( _nb.begin() + n );
    // Adjust neighbor entries of nodes
    for( size_t n2 = 0; n2 < nrNodes(); n2++ ) {
        for( size_t iter = 0; iter < nb(n2).size(); ) {
            Neighbor &m = nb(n2, iter);
            if( m.node == n ) {
                // delete this entry, because it points to the deleted node
                nb(n2).erase( nb(n2).begin() + iter );
            } else {
                // update this entry and the corresponding dual of the neighboring node
                if( m.node > n ) 
                    m.node--;
                nb( m.node, m.dual ).dual = iter;
                m.iter = iter++;
            }
        }
    }
}


void GraphAL::eraseEdge( size_t n1, size_t n2 ) {
    DAI_ASSERT( n1 < nrNodes() );
    DAI_ASSERT( n2 < nrNodes() );
    size_t iter;
    // Search for edge among neighbors of n1
    for( iter = 0; iter < nb(n1).size(); iter++ )
        if( nb(n1, iter).node == n2 ) {
            // Remove it
            nb(n1).erase( nb(n1).begin() + iter );
            break;
        }
    // Change the iter and dual values of the subsequent neighbors
    for( ; iter < nb(n1).size(); iter++ ) {
        Neighbor &m = nb( n1, iter );
        m.iter = iter;
        nb( m.node, m.dual ).dual = iter;
    }
    // Search for edge among neighbors of n2
    for( iter = 0; iter < nb(n2).size(); iter++ )
        if( nb(n2, iter).node == n1 ) {
            // Remove it
            nb(n2).erase( nb(n2).begin() + iter );
            break;
        }
    // Change the iter and node values of the subsequent neighbors
    for( ; iter < nb(n2).size(); iter++ ) {
        Neighbor &m = nb( n2, iter );
        m.iter = iter;
        nb( m.node, m.dual ).dual = iter;
    }
}


SmallSet<size_t> GraphAL::nbSet( size_t n ) const {
    SmallSet<size_t> result;
    foreach( const Neighbor &m, nb(n) )
        result |= m;
    return result;
}


bool GraphAL::isConnected() const {
    if( nrNodes() == 0 ) {
        return true;
    } else {
        std::vector<bool> incomponent( nrNodes(), false );

        incomponent[0] = true;
        bool found_new_nodes;
        do {
            found_new_nodes = false;

            // For all nodes, check if they are connected with the (growing) component
            for( size_t n1 = 0; n1 < nrNodes(); n1++ )
                if( !incomponent[n1] ) {
                    foreach( const Neighbor &n2, nb(n1) ) {
                        if( incomponent[n2] ) {
                            found_new_nodes = true;
                            incomponent[n1] = true;
                            break;
                        }
                    }
                }
        } while( found_new_nodes );

        // Check if there are remaining nodes (not in the component)
        bool all_connected = true;
        for( size_t n1 = 0; (n1 < nrNodes()) && all_connected; n1++ )
            if( !incomponent[n1] )
                all_connected = false;

        return all_connected;

        // BGL implementation is slower...
    /*  using namespace boost;
        typedef adjacency_list< vecS, vecS, undirectedS, property<vertex_distance_t, int> > boostGraphAL;
        typedef pair<size_t, size_t> E;

        // Copy graph structure into boostGraphAL object
        vector<E> edges;
        edges.reserve( nrEdges() );
        for( size_t n1 = 0; n1 < nrNodes(); n1++ )
            foreach( const Neighbor &n2, nb(n1) )
                if( n1 < n2 )
                    edges.push_back( E( n1, n2 ) );
        boostGraphAL g( edges.begin(), edges.end(), nrNodes() );

        // Construct connected components using Boost GraphAL Library
        std::vector<int> component( num_vertices( g ) );
        int num_comp = connected_components( g, make_iterator_property_map(component.begin(), get(vertex_index, g)) );

        return (num_comp == 1);
    */
    }
}


bool GraphAL::isTree() const {
    typedef vector<Edge> levelType; // first is node, second is its parent
    vector<levelType> levels;

    if( nrNodes() == 0 )
        return true;
    else {
        // start with root node 0
        levels.push_back( levelType( 1, Edge( 0, 0 ) ) );
        size_t treeSize = 1;
        bool foundCycle = false;
        do {
            levels.push_back( levelType() );
            const levelType &prevLevel = levels[levels.size() - 2];
            // build new level: add all neighbors of nodes in the previous level
            // (without backtracking), aborting if a cycle is detected
            for( size_t e = 0; e < prevLevel.size(); e++ ) {
                size_t n2 = prevLevel[e].first; // for all nodes n2 in the previous level
                foreach( const Neighbor &n1, nb(n2) ) { // for all neighbors n1 of n2
                    if( n1 != prevLevel[e].second ) { // no backtracking allowed
                        for( size_t l = 0; l < levels.size() && !foundCycle; l++ )
                            for( size_t f = 0; f < levels[l].size() && !foundCycle; f++ )
                                if( levels[l][f].first == n1 )
                                    // n1 has been visited before -> found a cycle
                                    foundCycle = true;
                        if( !foundCycle )
                            // add n1 (and its parent n2) to current level
                            levels.back().push_back( Edge( n1, n2 ) ); 
                    }
                    if( foundCycle )
                        break;
                }
                if( foundCycle )
                    break;
            }
            treeSize += levels.back().size();
        } while( (levels.back().size() != 0) && !foundCycle );
        if( treeSize == nrNodes() && !foundCycle )
            return true;
        else
            return false;
    }
}


void GraphAL::printDot( std::ostream& os ) const {
    os << "graph GraphAL {" << endl;
    os << "node[shape=circle,width=0.4,fixedsize=true];" << endl;
    for( size_t n = 0; n < nrNodes(); n++ )
        os << "\tx" << n << ";" << endl;
    for( size_t n1 = 0; n1 < nrNodes(); n1++ )
        foreach( const Neighbor &n2, nb(n1) )
            if( n1 < n2 )
                os << "\tx" << n1 << " -- x" << n2 << ";" << endl;
    os << "}" << endl;
}


void GraphAL::checkConsistency() const {
    size_t N = nrNodes();
    for( size_t n1 = 0; n1 < N; n1++ ) {
        size_t iter = 0;
        foreach( const Neighbor &n2, nb(n1) ) {
            DAI_ASSERT( n2.iter == iter );
            DAI_ASSERT( n2.node < N );
            DAI_ASSERT( n2.dual < nb(n2).size() );
            DAI_ASSERT( nb(n2, n2.dual) == n1 );
            iter++;
        }
    }
}


GraphAL createGraphFull( size_t N ) {
    GraphAL result( N );
    for( size_t i = 0; i < N; i++ )
        for( size_t j = i+1; j < N; j++ )
            result.addEdge( i, j, false );
    return result;
}


GraphAL createGraphGrid( size_t N1, size_t N2, bool periodic ) {
    GraphAL result( N1*N2 );
    if( N1 == 1 && N2 == 1 )
        return result;
    for( size_t i1 = 0; i1 < N1; i1++ )
        for( size_t i2 = 0; i2 < N2; i2++ ) {
            if( i1+1 < N1 || periodic )
                result.addEdge( i1*N2 + i2, ((i1+1)%N1)*N2 + i2, N1 <= 2 );
            if( i2+1 < N2 || periodic )
                result.addEdge( i1*N2 + i2, i1*N2 + ((i2+1)%N2), N2 <= 2 );
        }
    return result;
}


GraphAL createGraphGrid3D( size_t N1, size_t N2, size_t N3, bool periodic ) {
    GraphAL result( N1*N2*N3 );
    for( size_t i1 = 0; i1 < N1; i1++ )
        for( size_t i2 = 0; i2 < N2; i2++ )
            for( size_t i3 = 0; i3 < N3; i3++ ) {
                if( i1+1 < N1 || periodic )
                    result.addEdge( i1*N2*N3 + i2*N3 + i3, ((i1+1)%N1)*N2*N3 + i2*N3 + i3, N1 <= 2 );
                if( i2+1 < N2 || periodic )
                    result.addEdge( i1*N2*N3 + i2*N3 + i3, i1*N2*N3 + ((i2+1)%N2)*N3 + i3, N2 <= 2 );
                if( i3+1 < N3 || periodic )
                    result.addEdge( i1*N2*N3 + i2*N3 + i3, i1*N2*N3 + i2*N3 + ((i3+1)%N3), N3 <= 2 );
            }
    return result;
}


GraphAL createGraphLoop( size_t N ) {
    GraphAL result( N );
    for( size_t i = 0; i < N; i++ )
        result.addEdge( i, (i+1)%N, N <= 2 );
    return result;
}


GraphAL createGraphTree( size_t N ) {
    GraphAL result( N );
    for( size_t i = 1; i < N; i++ ) {
        size_t j = rnd_int( 0, i-1 );
        result.addEdge( i, j, false );
    }
    return result;
}


GraphAL createGraphRegular( size_t N, size_t d ) {
    DAI_ASSERT( (N * d) % 2 == 0 );
    DAI_ASSERT( d < N );

    GraphAL G( N );
    if( d > 0 ) {
        bool ready = false;
        size_t tries = 0;
        while( !ready ) {
            tries++;

            // Start with N*d points {0,1,...,N*d-1} (N*d even) in N groups.
            // Put U = {0,1,...,N*d-1}. (U denotes the set of unpaired points.)
            vector<size_t> U;
            U.reserve( N * d );
            for( size_t i = 0; i < N * d; i++ )
                U.push_back( i );

            // Repeat the following until no suitable pair can be found: Choose
            // two random points i and j in U, and if they are suitable, pair
            // i with j and delete i and j from U.
            G = GraphAL( N );
            bool finished = false;
            while( !finished ) {
                random_shuffle( U.begin(), U.end(), rnd );
                size_t i1, i2;
                bool suit_pair_found = false;
                for( i1 = 0; i1 < U.size()-1 && !suit_pair_found; i1++ )
                    for( i2 = i1+1; i2 < U.size() && !suit_pair_found; i2++ )
                        if( ((U[i1] / d) != (U[i2] / d)) && !G.hasEdge( U[i1] / d, U[i2] / d ) ) {
                            // they are suitable
                            suit_pair_found = true;
                            G.addEdge( U[i1] / d, U[i2] / d, false );
                            U.erase( U.begin() + i2 );  // first remove largest
                            U.erase( U.begin() + i1 );  // remove smallest
                        }
                if( !suit_pair_found || U.empty() )
                    finished = true;
            }

            if( U.empty() ) {
                // G is a graph with edge from vertex r to vertex s if and only if
                // there is a pair containing points in the r'th and s'th groups.
                // If G is d-regular, output, otherwise return to Step 1.
                ready = true;
                for( size_t n = 0; n < N; n++ )
                    if( G.nb(n).size() != d ) {
                        ready = false;
                        break;
                    }
            } else
                ready = false;
        }
    }

    return G;
}


} // end of namespace dai
