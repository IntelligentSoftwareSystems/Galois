/*  This file is part of libDAI - http://www.libdai.org/
 *
 *  Copyright (c) 2006-2011, The libDAI authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
 */


#include <set>
#include <vector>
#include <iostream>
#include <dai/varset.h>
#include <dai/clustergraph.h>


namespace dai {


using namespace std;


ClusterGraph::ClusterGraph( const std::vector<VarSet> & cls ) : _G(), _vars(), _clusters() {
    // construct vars, clusters and edge list
    vector<Edge> edges;
    foreach( const VarSet &cl, cls ) {
        if( find( clusters().begin(), clusters().end(), cl ) == clusters().end() ) {
            // add cluster
            size_t n2 = nrClusters();
            _clusters.push_back( cl );
            for( VarSet::const_iterator n = cl.begin(); n != cl.end(); n++ ) {
                size_t n1 = find( vars().begin(), vars().end(), *n ) - vars().begin();
                if( n1 == nrVars() )
                    // add variable
                    _vars.push_back( *n );
                edges.push_back( Edge( n1, n2 ) );
            }
        } // disregard duplicate clusters
    }

    // Create bipartite graph
    _G.construct( nrVars(), nrClusters(), edges.begin(), edges.end() );
}


ClusterGraph::ClusterGraph( const FactorGraph& fg, bool onlyMaximal ) : _G( fg.nrVars(), 0 ), _vars(), _clusters() {
    // copy variables
    _vars.reserve( fg.nrVars() );
    for( size_t i = 0; i < fg.nrVars(); i++ )
        _vars.push_back( fg.var(i) );

    if( onlyMaximal ) {
        for( size_t I = 0; I < fg.nrFactors(); I++ )
            if( fg.isMaximal( I ) ) {
                _clusters.push_back( fg.factor(I).vars() );
                size_t clind = _G.addNode2();
                foreach( const Neighbor &i, fg.nbF(I) )
                    _G.addEdge( i, clind, true );
            }
    } else {
        // copy clusters
        _clusters.reserve( fg.nrFactors() );
        for( size_t I = 0; I < fg.nrFactors(); I++ )
            _clusters.push_back( fg.factor(I).vars() );
        // copy bipartite graph
        _G = fg.bipGraph();
    }
}


size_t sequentialVariableElimination::operator()( const ClusterGraph &cl, const std::set<size_t> &/*remainingVars*/ ) {
    return cl.findVar( seq.at(i++) );
}


size_t greedyVariableElimination::operator()( const ClusterGraph &cl, const std::set<size_t> &remainingVars ) {
    set<size_t>::const_iterator lowest = remainingVars.end();
    size_t lowest_cost = -1UL;
    for( set<size_t>::const_iterator i = remainingVars.begin(); i != remainingVars.end(); i++ ) {
        size_t cost = heuristic( cl, *i );
        if( lowest == remainingVars.end() || lowest_cost > cost ) {
            lowest = i;
            lowest_cost = cost;
        }
    }
    return *lowest;
}


size_t eliminationCost_MinNeighbors( const ClusterGraph &cl, size_t i ) {
    return cl.bipGraph().delta1( i ).size();
}


size_t eliminationCost_MinWeight( const ClusterGraph &cl, size_t i ) {
    SmallSet<size_t> id_n = cl.bipGraph().delta1( i );
    
    size_t cost = 1;
    for( SmallSet<size_t>::const_iterator it = id_n.begin(); it != id_n.end(); it++ )
        cost *= cl.vars()[*it].states();

    return cost;
}


size_t eliminationCost_MinFill( const ClusterGraph &cl, size_t i ) {
    SmallSet<size_t> id_n = cl.bipGraph().delta1( i );

    size_t cost = 0;
    // for each unordered pair {i1,i2} adjacent to n
    for( SmallSet<size_t>::const_iterator it1 = id_n.begin(); it1 != id_n.end(); it1++ )
        for( SmallSet<size_t>::const_iterator it2 = it1; it2 != id_n.end(); it2++ )
            if( it1 != it2 ) {
                // if i1 and i2 are not adjacent, eliminating n would make them adjacent
                if( !cl.adj(*it1, *it2) )
                    cost++;
            }

    return cost;
}


size_t eliminationCost_WeightedMinFill( const ClusterGraph &cl, size_t i ) {
    SmallSet<size_t> id_n = cl.bipGraph().delta1( i );

    size_t cost = 0;
    // for each unordered pair {i1,i2} adjacent to n
    for( SmallSet<size_t>::const_iterator it1 = id_n.begin(); it1 != id_n.end(); it1++ )
        for( SmallSet<size_t>::const_iterator it2 = it1; it2 != id_n.end(); it2++ )
            if( it1 != it2 ) {
                // if i1 and i2 are not adjacent, eliminating n would make them adjacent
                if( !cl.adj(*it1, *it2) )
                    cost += cl.vars()[*it1].states() * cl.vars()[*it2].states();
            }

    return cost;
}


} // end of namespace dai
