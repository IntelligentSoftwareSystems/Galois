/*  This file is part of libDAI - http://www.libdai.org/
 *
 *  Copyright (c) 2006-2011, The libDAI authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
 */


#include <dai/alldai.h>


namespace dai {


using namespace std;


void DecMAP::setProperties( const PropertySet &opts ) {
    DAI_ASSERT( opts.hasKey("ianame") );
    DAI_ASSERT( opts.hasKey("iaopts") );

    props.ianame = opts.getStringAs<string>("ianame");
    props.iaopts = opts.getStringAs<PropertySet>("iaopts");
    if( opts.hasKey("verbose") )
        props.verbose = opts.getStringAs<size_t>("verbose");
    else
        props.verbose = 0;
    if( opts.hasKey("reinit") )
        props.reinit = opts.getStringAs<bool>("reinit");
    else
        props.reinit = true;
}


PropertySet DecMAP::getProperties() const {
    PropertySet opts;
    opts.set( "verbose", props.verbose );
    opts.set( "reinit", props.reinit );
    opts.set( "ianame", props.ianame );
    opts.set( "iaopts", props.iaopts );
    return opts;
}


string DecMAP::printProperties() const {
    stringstream s( stringstream::out );
    s << "[";
    s << "verbose=" << props.verbose << ",";
    s << "reinit=" << props.reinit << ",";
    s << "ianame=" << props.ianame << ",";
    s << "iaopts=" << props.iaopts << "]";
    return s.str();
}


DecMAP::DecMAP( const FactorGraph& fg, const PropertySet& opts ) : DAIAlgFG(fg), _state(), _logp(), _maxdiff(), _iters(), props() {
    setProperties( opts );

    _state = vector<size_t>( nrVars(), 0 );
    _logp = -INFINITY;
}


Factor DecMAP::belief( const VarSet& vs ) const {
    if( vs.size() == 0 )
        return Factor();
    else {
        map<Var, size_t> state;
        for( VarSet::const_iterator v = vs.begin(); v != vs.end(); v++ )
            state[*v] = _state[findVar(*v)];
        return createFactorDelta( vs, calcLinearState( vs, state ) );
    }
}


Factor DecMAP::beliefV( size_t i ) const {
    return createFactorDelta( var(i), _state[i] );
}


vector<Factor> DecMAP::beliefs() const {
    vector<Factor> result;
    for( size_t i = 0; i < nrVars(); ++i )
        result.push_back( beliefV(i) );
    for( size_t I = 0; I < nrFactors(); ++I )
        result.push_back( beliefF(I) );
    return result;
}


Real DecMAP::run() {
    if( props.verbose >= 1 )
        cerr << "Starting " << identify() << "...";
    if( props.verbose >= 2 )
        cerr << endl;

    // the variables which have not been clamped yet
    SmallSet<size_t> freeVars;
    for( size_t i = 0; i < nrVars(); i++ )
        freeVars |= i;

    // prepare the inference algorithm object
    InfAlg *clamped = newInfAlg( props.ianame, fg(), props.iaopts );

    // decimate until no free variables remain
    while( freeVars.size() ) {
        Real md = clamped->run();
        if( md > _maxdiff )
            _maxdiff = md;
        _iters += clamped->Iterations();

        // store the variables that need initialization
        VarSet varsToInit;
        SmallSet<size_t> varsToClamp;

        // schedule clamping for the free variables with zero entropy
        for( SmallSet<size_t>::const_iterator it = freeVars.begin(); it != freeVars.end(); ) {
            if( clamped->beliefV( *it ).entropy() == 0.0 ) {
                // this variable should be clamped
                varsToInit |= var( *it );
                varsToClamp |= *it;
                _state[*it] = clamped->beliefV( *it ).p().argmax().first;
                freeVars.erase( *it );
            } else
                it++;
        }

        // find the free factor with lowest entropy
        size_t bestI = 0;
        Real bestEnt = INFINITY;
        for( size_t I = 0; I < nrFactors(); I++ ) {
            // check if the factor is still free
            if( freeVars.intersects( bipGraph().nb2Set(I) ) ) {
                Real EntI = clamped->beliefF(I).entropy();
                if( EntI < bestEnt ) {
                    bestI = I;
                    bestEnt = EntI;
                }
            }
        }

        // schedule clamping for the factor with lowest entropy
        vector<size_t> Istate(1,0);
        Istate[0] = clamped->beliefF(bestI).p().argmax().first;
        map<Var, size_t> Istatemap = calcState( factor(bestI).vars(), Istate[0] );
        foreach( size_t i, bipGraph().nb2Set(bestI) & freeVars ) {
            varsToInit |= var(i);
            varsToClamp |= i;
            _state[i] = Istatemap[var(i)];
            freeVars.erase(i);
        }

        // clamp all variables scheduled for clamping
        foreach( size_t i, varsToClamp )
            clamped->clamp( i, _state[i], false );

        // initialize clamped for the next run
        if( props.reinit )
            clamped->init();
        else
            clamped->init( varsToInit );
    }

    // calculate MAP state
    map<Var, size_t> state;
    for( size_t i = 0; i < nrVars(); i++ )
        state[var(i)] = _state[i];
    _logp = 0.0;
    for( size_t I = 0; I < nrFactors(); I++ )
        _logp += dai::log( factor(I)[calcLinearState( factor(I).vars(), state )] );

    // clean up
    delete clamped;

    return _maxdiff;
}


} // end of namespace dai
