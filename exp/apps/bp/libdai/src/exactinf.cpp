/*  This file is part of libDAI - http://www.libdai.org/
 *
 *  Copyright (c) 2006-2011, The libDAI authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
 */


#include <dai/exactinf.h>
#include <sstream>


namespace dai {


using namespace std;


void ExactInf::setProperties( const PropertySet &opts ) {
    if( opts.hasKey("verbose") )
        props.verbose = opts.getStringAs<size_t>("verbose");
    else
        props.verbose = 0;
}


PropertySet ExactInf::getProperties() const {
    PropertySet opts;
    opts.set( "verbose", props.verbose );
    return opts;
}


string ExactInf::printProperties() const {
    stringstream s( stringstream::out );
    s << "[";
    s << "verbose=" << props.verbose << "]";
    return s.str();
}


void ExactInf::construct() {
    // clear variable beliefs and reserve space
    _beliefsV.clear();
    _beliefsV.reserve( nrVars() );
    for( size_t i = 0; i < nrVars(); i++ )
        _beliefsV.push_back( Factor( var(i) ) );

    // clear factor beliefs and reserve space
    _beliefsF.clear();
    _beliefsF.reserve( nrFactors() );
    for( size_t I = 0; I < nrFactors(); I++ )
        _beliefsF.push_back( Factor( factor(I).vars() ) );
}


void ExactInf::init() {
    for( size_t i = 0; i < nrVars(); i++ )
        _beliefsV[i].fill( 1.0 );
    for( size_t I = 0; I < nrFactors(); I++ )
        _beliefsF[I].fill( 1.0 );
}


Real ExactInf::run() {
    if( props.verbose >= 1 )
        cerr << "Starting " << identify() << "...";

    Factor P;
    for( size_t I = 0; I < nrFactors(); I++ )
        P *= factor(I);

    Real Z = P.sum();
    _logZ = std::log(Z);
    for( size_t i = 0; i < nrVars(); i++ )
        _beliefsV[i] = P.marginal(var(i));
    for( size_t I = 0; I < nrFactors(); I++ )
        _beliefsF[I] = P.marginal(factor(I).vars());

    if( props.verbose >= 1 )
        cerr << "finished" << endl;

    return 0.0;
}


Factor ExactInf::calcMarginal( const VarSet &vs ) const {
    Factor P;
    for( size_t I = 0; I < nrFactors(); I++ )
        P *= factor(I);
    return P.marginal( vs, true );
}

        
std::vector<std::size_t> ExactInf::findMaximum() const {
    Factor P;
    for( size_t I = 0; I < nrFactors(); I++ )
        P *= factor(I);
    size_t linearState = P.p().argmax().first;

    // convert to state
    map<Var, size_t> state = calcState( P.vars(), linearState );

    // convert to desired output data structure
    vector<size_t> mapState;
    mapState.reserve( nrVars() );
    for( size_t i = 0; i < nrVars(); i++ )
        mapState.push_back( state[var(i)] );

    return mapState;
}


vector<Factor> ExactInf::beliefs() const {
    vector<Factor> result = _beliefsV;
    result.insert( result.end(), _beliefsF.begin(), _beliefsF.end() );
    return result;
}


Factor ExactInf::belief( const VarSet &ns ) const {
    if( ns.size() == 0 )
        return Factor();
    else if( ns.size() == 1 ) {
        return beliefV( findVar( *(ns.begin()) ) );
    } else {
        size_t I;
        for( I = 0; I < nrFactors(); I++ )
            if( factor(I).vars() >> ns )
                break;
        if( I == nrFactors() )
            DAI_THROW(BELIEF_NOT_AVAILABLE);
        return beliefF(I).marginal(ns);
    }
}


} // end of namespace dai
