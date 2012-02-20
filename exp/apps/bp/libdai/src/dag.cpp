/*  This file is part of libDAI - http://www.libdai.org/
 *
 *  Copyright (c) 2006-2011, The libDAI authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
 */


#include <dai/dag.h>


namespace dai {


using namespace std;


DAG& DAG::addEdge( size_t n1, size_t n2, bool check ) {
    DAI_ASSERT( n1 < nrNodes() );
    DAI_ASSERT( n2 < nrNodes() );
    bool allowed = true;
    if( check ) {
        // Check whether the edge already exists
        foreach( const Neighbor& n, ch(n1) )
            if( n == n2 ) {
                allowed = false;
                break;
            }
        // Check whether there exists a directed path from n2 to n1 already
        if( allowed && existsDirectedPath( n2, n1 ) )
            allowed = false;
    }
    if( allowed && n1 != n2 ) { // Add edge
        Neighbor ch_1( ch(n1).size(), n2, pa(n2).size() );
        Neighbor pa_2( ch_1.dual, n1, ch_1.iter );
        ch(n1).push_back( ch_1 );
        pa(n2).push_back( pa_2 );
    }
    return *this;
}


void DAG::eraseNode( size_t n ) {
    DAI_ASSERT( n < nrNodes() );
    // Erase parents entry of node n
    _pa.erase( _pa.begin() + n );
    // Erase children entry of node n
    _ch.erase( _ch.begin() + n );
    // Adjust parents and children entries of remaining nodes
    for( size_t m = 0; m < nrNodes(); m++ ) {
        // Adjust parents entries of node m
        for( size_t iter = 0; iter < pa(m).size(); ) {
            Neighbor &p = pa(m, iter);
            if( p.node == n ) {
                // delete this entry, because it points to the deleted node
                pa(m).erase( pa(m).begin() + iter );
            } else {
                // update this entry and the corresponding dual of the child node
                if( p.node > n )
                    p.node--;
                ch( p.node, p.dual ).dual = iter;
                p.iter = iter++;
            }
        }
        // Adjust children entries of node m
        for( size_t iter = 0; iter < ch(m).size(); ) {
            Neighbor &c = ch(m, iter);
            if( c.node == n ) {
                // delete this entry, because it points to the deleted node
                ch(m).erase( ch(m).begin() + iter );
            } else {
                if( c.node > n )
                    c.node--;
                // update this entry and the corresponding dual of the child node
                pa( c.node, c.dual ).dual = iter;
                c.iter = iter++;
            }
        }
    }
}


void DAG::eraseEdge( size_t n1, size_t n2 ) {
    DAI_ASSERT( n1 < nrNodes() );
    DAI_ASSERT( n2 < nrNodes() );
    size_t iter;
    // Search for edge among children of n1
    for( iter = 0; iter < ch(n1).size(); iter++ )
        if( ch(n1, iter).node == n2 ) {
            // Remove it
            ch(n1).erase( ch(n1).begin() + iter );
            break;
        }
    // Change the iter and dual values of the subsequent children
    for( ; iter < ch(n1).size(); iter++ ) {
        Neighbor &m = ch( n1, iter );
        m.iter = iter;
        pa( m.node, m.dual ).dual = iter;
    }
    // Search for edge among parents of n2
    for( iter = 0; iter < pa(n2).size(); iter++ )
        if( pa(n2, iter).node == n1 ) {
            // Remove it
            pa(n2).erase( pa(n2).begin() + iter );
            break;
        }
    // Change the iter and node values of the subsequent parents
    for( ; iter < pa(n2).size(); iter++ ) {
        Neighbor &m = pa( n2, iter );
        m.iter = iter;
        ch( m.node, m.dual ).dual = iter;
    }
}


SmallSet<size_t> DAG::paSet( size_t n ) const {
    SmallSet<size_t> result;
    foreach( const Neighbor &p, pa(n) )
        result |= p;
    return result;
}


SmallSet<size_t> DAG::chSet( size_t n ) const {
    SmallSet<size_t> result;
    foreach( const Neighbor &c, ch(n) )
        result |= c;
    return result;
}


/// Returns the set of ancestors of node \a n, i.e., all nodes \a m such that there exists a directed path from \a m to \a n (excluding \a n itself)
std::set<size_t> DAG::ancestors( size_t n ) const {
    set<size_t> anc;

    set<size_t> curParents;
    curParents.insert( n );
    while( curParents.size() ) {
        size_t node = *(curParents.begin());
        // Add node to ancestors
        if( node != n )
            anc.insert( node );
        // Add all parents of node to curParents
        foreach( const Neighbor& p, pa(node) )
            curParents.insert( p.node );
        // Erase node from curParents
        curParents.erase( node );
    }

    return anc;
}


/// Returns the set of descendants of node \a n, i.e., all nodes \a m such that there exists a directed path from \a n to \a m (excluding \a n itself)
std::set<size_t> DAG::descendants( size_t n ) const {
    set<size_t> desc;

    set<size_t> curChildren;
    curChildren.insert( n );
    while( curChildren.size() ) {
        size_t node = *(curChildren.begin());
        // Add node to descendants
        if( node != n )
            desc.insert( node );
        // Add all children of node to curChildren
        foreach( const Neighbor& c, ch(node) )
            curChildren.insert( c.node );
        // Erase node from curChildren
        curChildren.erase( node );
    }

    return desc;
}


bool DAG::existsDirectedPath( size_t n1, size_t n2 ) const {
    set<size_t> curChildren;

    curChildren.insert( n1 );
    while( curChildren.size() ) {
        size_t node = *(curChildren.begin());
        // If we reached n2, we found a directed path from n1 to n2
        if( node == n2 )
            return true;
        // Add all children of node to curChildren
        foreach( const Neighbor& c, ch(node) )
            curChildren.insert( c.node );
        // Erase node from curChildren
        curChildren.erase( node );
    }
    return false;
}


bool DAG::isConnected() const {
    if( nrNodes() == 0 ) {
        return true;
    } else {
        std::vector<bool> incomponent( nrNodes(), false );

        incomponent[0] = true;
        bool found_new_nodes;
        do {
            found_new_nodes = false;

            // For all nodes, check if they are connected with the (growing) component
            for( size_t n1 = 0; n1 < nrNodes(); n1++ ) {
                if( !incomponent[n1] ) {
                    foreach( const Neighbor &n2, pa(n1) )
                        if( incomponent[n2] ) {
                            found_new_nodes = true;
                            incomponent[n1] = true;
                            break;
                        }
                }
                if( !incomponent[n1] ) {
                    foreach( const Neighbor &n2, ch(n1) )
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
    }
}


void DAG::printDot( std::ostream& os ) const {
    os << "digraph DAG {" << endl;
    os << "node[shape=circle,width=0.4,fixedsize=true];" << endl;
    for( size_t n = 0; n < nrNodes(); n++ )
        os << "\tx" << n << ";" << endl;
    for( size_t n1 = 0; n1 < nrNodes(); n1++ )
        foreach( const Neighbor &n2, ch(n1) )
            os << "\tx" << n1 << " -> x" << n2 << ";" << endl;
    os << "}" << endl;
}


void DAG::checkConsistency() const {
    size_t N = nrNodes();
    for( size_t n1 = 0; n1 < N; n1++ ) {
        size_t iter = 0;
        foreach( const Neighbor &n2, pa(n1) ) {
            DAI_ASSERT( n2.iter == iter );
            DAI_ASSERT( n2.node < N );
            DAI_ASSERT( n2.dual < ch(n2).size() );
            DAI_ASSERT( ch(n2, n2.dual) == n1 );
            iter++;
        }
        iter = 0;
        foreach( const Neighbor &n2, ch(n1) ) {
            DAI_ASSERT( n2.iter == iter );
            DAI_ASSERT( n2.node < N );
            DAI_ASSERT( n2.dual < pa(n2).size() );
            DAI_ASSERT( pa(n2, n2.dual) == n1 );
            iter++;
        }
    }
    // Check acyclicity
    for( size_t n1 = 0; n1 < N; n1++ )
        foreach( const Neighbor& n2, ch(n1) )
            DAI_ASSERT( !existsDirectedPath( n2, n1 ) );
}


} // end of namespace dai
