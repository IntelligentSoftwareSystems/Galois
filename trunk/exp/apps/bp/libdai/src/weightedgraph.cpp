/*  This file is part of libDAI - http://www.libdai.org/
 *
 *  Copyright (c) 2006-2011, The libDAI authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
 */


#include <algorithm>
#include <dai/weightedgraph.h>
#include <dai/util.h>
#include <dai/exceptions.h>


namespace dai {


using namespace std;


RootedTree::RootedTree( const GraphEL &T, size_t Root ) {
    if( T.size() != 0 ) {
        // Make a copy
        GraphEL Gr = T;

        // Nodes in the tree
        set<size_t> nodes;

        // Check whether the root is in the tree
        bool valid = false;
        for( GraphEL::iterator e = Gr.begin(); e != Gr.end() && !valid; e++ )
            if( e->first == Root || e->second == Root )
                valid = true;
        if( !valid )
            DAI_THROWE(RUNTIME_ERROR,"Graph does not contain specified root.");

        // Start with the root
        nodes.insert( Root );

        // Keep adding edges until done
        bool done = false;
        while( !done ) {
            bool changed = false;
            for( GraphEL::iterator e = Gr.begin(); e != Gr.end(); ) {
                bool e1_in_nodes = nodes.count( e->first );
                bool e2_in_nodes = nodes.count( e->second );
                if( e1_in_nodes && e2_in_nodes )
                    DAI_THROWE(RUNTIME_ERROR,"Graph is not acyclic.");
                if( e1_in_nodes ) {
                    // Add directed edge, pointing away from the root
                    push_back( DEdge( e->first, e->second ) );
                    nodes.insert( e->second );
                    // Erase the edge
                    Gr.erase( e++ );
                    changed = true;
                } else if( e2_in_nodes ) {
                    // Add directed edge, pointing away from the root
                    push_back( DEdge( e->second, e->first ) );
                    nodes.insert( e->first );
                    // Erase the edge
                    Gr.erase( e++ );
                    changed = true;
                } else
                    e++;
            }
            if( Gr.empty() )
                done = true;
            if( !changed && !done )
                DAI_THROWE(RUNTIME_ERROR,"Graph is not connected.");
        }
    }
}


} // end of namespace dai
