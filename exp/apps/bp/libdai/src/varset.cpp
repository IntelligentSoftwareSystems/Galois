/*  This file is part of libDAI - http://www.libdai.org/
 *
 *  Copyright (c) 2006-2011, The libDAI authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
 */


#include <dai/varset.h>


namespace dai {


using namespace std;


size_t calcLinearState( const VarSet &vs, const std::map<Var, size_t> &state ) {
    size_t prod = 1;
    size_t st = 0;
    for( VarSet::const_iterator v = vs.begin(); v != vs.end(); v++ ) {
        std::map<Var, size_t>::const_iterator m = state.find( *v );
        if( m != state.end() )
            st += prod * m->second;
        prod *= v->states();
    }
    return st;
}


std::map<Var, size_t> calcState( const VarSet &vs, size_t linearState ) {
    std::map<Var, size_t> state;
    for( VarSet::const_iterator v = vs.begin(); v != vs.end(); v++ ) {
        state[*v] = linearState % v->states();
        linearState /= v->states();
    }
    DAI_ASSERT( linearState == 0 );
    return state;
}


} // end of namespace dai
