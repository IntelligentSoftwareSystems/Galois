/*  This file is part of libDAI - http://www.libdai.org/
 *
 *  Copyright (c) 2006-2011, The libDAI authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
 */


#include <iostream>
#include <algorithm>
#include <map>
#include <set>
#include <dai/lc.h>
#include <dai/util.h>
#include <dai/alldai.h>


namespace dai {


using namespace std;


void LC::setProperties( const PropertySet &opts ) {
    DAI_ASSERT( opts.hasKey("tol") );
    DAI_ASSERT( opts.hasKey("maxiter") );
    DAI_ASSERT( opts.hasKey("cavity") );
    DAI_ASSERT( opts.hasKey("updates") );

    props.tol = opts.getStringAs<Real>("tol");
    props.maxiter = opts.getStringAs<size_t>("maxiter");
    props.cavity = opts.getStringAs<Properties::CavityType>("cavity");
    props.updates = opts.getStringAs<Properties::UpdateType>("updates");
    if( opts.hasKey("verbose") )
        props.verbose = opts.getStringAs<size_t>("verbose");
    else
        props.verbose = 0;
    if( opts.hasKey("cavainame") )
        props.cavainame = opts.getStringAs<string>("cavainame");
    if( opts.hasKey("cavaiopts") )
        props.cavaiopts = opts.getStringAs<PropertySet>("cavaiopts");
    if( opts.hasKey("reinit") )
        props.reinit = opts.getStringAs<bool>("reinit");
    if( opts.hasKey("damping") )
        props.damping = opts.getStringAs<Real>("damping");
    else
        props.damping = 0.0;
}


PropertySet LC::getProperties() const {
    PropertySet opts;
    opts.set( "tol", props.tol );
    opts.set( "maxiter", props.maxiter );
    opts.set( "verbose", props.verbose );
    opts.set( "cavity", props.cavity );
    opts.set( "updates", props.updates );
    opts.set( "cavainame", props.cavainame );
    opts.set( "cavaiopts", props.cavaiopts );
    opts.set( "reinit", props.reinit );
    opts.set( "damping", props.damping );
    return opts;
}


string LC::printProperties() const {
    stringstream s( stringstream::out );
    s << "[";
    s << "tol=" << props.tol << ",";
    s << "maxiter=" << props.maxiter << ",";
    s << "verbose=" << props.verbose << ",";
    s << "cavity=" << props.cavity << ",";
    s << "updates=" << props.updates << ",";
    s << "cavainame=" << props.cavainame << ",";
    s << "cavaiopts=" << props.cavaiopts << ",";
    s << "reinit=" << props.reinit << ",";
    s << "damping=" << props.damping << "]";
    return s.str();
}


LC::LC( const FactorGraph & fg, const PropertySet &opts ) : DAIAlgFG(fg), _pancakes(), _cavitydists(), _phis(), _beliefs(), _maxdiff(0.0), _iters(0), props() {
    setProperties( opts );

    // create pancakes
    _pancakes.resize( nrVars() );

    // create cavitydists
    for( size_t i=0; i < nrVars(); i++ )
        _cavitydists.push_back(Factor( delta(i) ));

    // create phis
    _phis.reserve( nrVars() );
    for( size_t i = 0; i < nrVars(); i++ ) {
        _phis.push_back( vector<Factor>() );
        _phis[i].reserve( nbV(i).size() );
        foreach( const Neighbor &I, nbV(i) )
            _phis[i].push_back( Factor( factor(I).vars() / var(i) ) );
    }

    // create beliefs
    _beliefs.reserve( nrVars() );
    for( size_t i=0; i < nrVars(); i++ )
        _beliefs.push_back(Factor(var(i)));
}


void LC::CalcBelief (size_t i) {
    _beliefs[i] = _pancakes[i].marginal(var(i));
}


Factor LC::belief (const VarSet &ns) const {
    if( ns.size() == 0 )
        return Factor();
    else if( ns.size() == 1 )
        return beliefV( findVar( *(ns.begin()) ) );
    else {
        DAI_THROW(BELIEF_NOT_AVAILABLE);
        return Factor();
    }
}


Real LC::CalcCavityDist (size_t i, const std::string &name, const PropertySet &opts) {
    Factor Bi;
    Real maxdiff = 0;

    if( props.verbose >= 2 )
        cerr << "Initing cavity " << var(i) << "(" << delta(i).size() << " vars, " << delta(i).nrStates() << " states)" << endl;

    if( props.cavity == Properties::CavityType::UNIFORM )
        Bi = Factor(delta(i));
    else {
        InfAlg *cav = newInfAlg( name, *this, opts );
        cav->makeCavity( i );

        if( props.cavity == Properties::CavityType::FULL )
            Bi = calcMarginal( *cav, cav->fg().delta(i), props.reinit );
        else if( props.cavity == Properties::CavityType::PAIR ) {
            vector<Factor> pairbeliefs = calcPairBeliefs( *cav, cav->fg().delta(i), props.reinit, false );
            for( size_t ij = 0; ij < pairbeliefs.size(); ij++ )
                Bi *= pairbeliefs[ij];
        } else if( props.cavity == Properties::CavityType::PAIR2 ) {
            vector<Factor> pairbeliefs = calcPairBeliefs( *cav, cav->fg().delta(i), props.reinit, true );
            for( size_t ij = 0; ij < pairbeliefs.size(); ij++ )
                Bi *= pairbeliefs[ij];
        }
        maxdiff = cav->maxDiff();
        delete cav;
    }
    Bi.normalize();
    _cavitydists[i] = Bi;

    return maxdiff;
}


Real LC::InitCavityDists( const std::string &name, const PropertySet &opts ) {
    double tic = toc();

    if( props.verbose >= 1 ) {
        cerr << this->name() << "::InitCavityDists:  ";
        if( props.cavity == Properties::CavityType::UNIFORM )
            cerr << "Using uniform initial cavity distributions" << endl;
        else if( props.cavity == Properties::CavityType::FULL )
            cerr << "Using full " << name << opts << "...";
        else if( props.cavity == Properties::CavityType::PAIR )
            cerr << "Using pairwise " << name << opts << "...";
        else if( props.cavity == Properties::CavityType::PAIR2 )
            cerr << "Using pairwise(new) " << name << opts << "...";
    }

    Real maxdiff = 0.0;
    for( size_t i = 0; i < nrVars(); i++ ) {
        Real md = CalcCavityDist(i, name, opts);
        if( md > maxdiff )
            maxdiff = md;
    }

    if( props.verbose >= 1 ) {
        cerr << this->name() << "::InitCavityDists used " << toc() - tic << " seconds." << endl;
    }

    return maxdiff;
}


long LC::SetCavityDists( std::vector<Factor> &Q ) {
    if( props.verbose >= 1 )
        cerr << name() << "::SetCavityDists:  Setting initial cavity distributions" << endl;
    if( Q.size() != nrVars() )
        return -1;
    for( size_t i = 0; i < nrVars(); i++ ) {
        if( _cavitydists[i].vars() != Q[i].vars() ) {
            return i+1;
        } else
            _cavitydists[i] = Q[i];
    }
    return 0;
}


void LC::init() {
    for( size_t i = 0; i < nrVars(); ++i )
        foreach( const Neighbor &I, nbV(i) )
            if( props.updates == Properties::UpdateType::SEQRND )
                _phis[i][I.iter].randomize();
            else
                _phis[i][I.iter].fill(1.0);
}


Factor LC::NewPancake (size_t i, size_t _I, bool & hasNaNs) {
    size_t I = nbV(i)[_I];
    Factor piet = _pancakes[i];

    // recalculate _pancake[i]
    VarSet Ivars = factor(I).vars();
    Factor A_I;
    for( VarSet::const_iterator k = Ivars.begin(); k != Ivars.end(); k++ )
        if( var(i) != *k )
            A_I *= (_pancakes[findVar(*k)] * factor(I).inverse()).marginal( Ivars / var(i), false );
    if( Ivars.size() > 1 )
        A_I ^= (1.0 / (Ivars.size() - 1));
    Factor A_Ii = (_pancakes[i] * factor(I).inverse() * _phis[i][_I].inverse()).marginal( Ivars / var(i), false );
    Factor quot = A_I / A_Ii;
    if( props.damping != 0.0 )
        quot = (quot^(1.0 - props.damping)) * (_phis[i][_I]^props.damping);

    piet *= quot / _phis[i][_I].normalized();
    _phis[i][_I] = quot.normalized();

    piet.normalize();

    if( piet.hasNaNs() ) {
        cerr << name() << "::NewPancake(" << i << ", " << _I << "):  has NaNs!" << endl;
        hasNaNs = true;
    }

    return piet;
}


Real LC::run() {
    if( props.verbose >= 1 )
        cerr << "Starting " << identify() << "...";
    if( props.verbose >= 2 )
        cerr << endl;

    double tic = toc();

    Real md = InitCavityDists( props.cavainame, props.cavaiopts );
    if( md > _maxdiff )
        _maxdiff = md;

    for( size_t i = 0; i < nrVars(); i++ ) {
        _pancakes[i] = _cavitydists[i];

        foreach( const Neighbor &I, nbV(i) ) {
            _pancakes[i] *= factor(I);
            if( props.updates == Properties::UpdateType::SEQRND )
              _pancakes[i] *= _phis[i][I.iter];
        }

        _pancakes[i].normalize();

        CalcBelief(i);
    }

    vector<Factor> oldBeliefsV;
    for( size_t i = 0; i < nrVars(); i++ )
        oldBeliefsV.push_back( beliefV(i) );

    bool hasNaNs = false;
    for( size_t i=0; i < nrVars(); i++ )
        if( _pancakes[i].hasNaNs() ) {
            hasNaNs = true;
            break;
        }
    if( hasNaNs ) {
        cerr << name() << "::run:  initial _pancakes has NaNs!" << endl;
        return 1.0;
    }

    size_t nredges = nrEdges();
    vector<Edge> update_seq;
    update_seq.reserve( nredges );
    for( size_t i = 0; i < nrVars(); ++i )
        foreach( const Neighbor &I, nbV(i) )
            update_seq.push_back( Edge( i, I.iter ) );

    // do several passes over the network until maximum number of iterations has
    // been reached or until the maximum belief difference is smaller than tolerance
    Real maxDiff = INFINITY;
    for( _iters = 0; _iters < props.maxiter && maxDiff > props.tol; _iters++ ) {
        // Sequential updates
        if( props.updates == Properties::UpdateType::SEQRND )
            random_shuffle( update_seq.begin(), update_seq.end(), rnd );

        for( size_t t=0; t < nredges; t++ ) {
            size_t i = update_seq[t].first;
            size_t _I = update_seq[t].second;
            _pancakes[i] = NewPancake( i, _I, hasNaNs);
            if( hasNaNs )
                return 1.0;
            CalcBelief( i );
        }

        // compare new beliefs with old ones
        maxDiff = -INFINITY;
        for( size_t i = 0; i < nrVars(); i++ ) {
            maxDiff = std::max( maxDiff, dist( beliefV(i), oldBeliefsV[i], DISTLINF ) );
            oldBeliefsV[i] = beliefV(i);
        }

        if( props.verbose >= 3 )
            cerr << name() << "::run:  maxdiff " << maxDiff << " after " << _iters+1 << " passes" << endl;
    }

    if( maxDiff > _maxdiff )
        _maxdiff = maxDiff;

    if( props.verbose >= 1 ) {
        if( maxDiff > props.tol ) {
            if( props.verbose == 1 )
                cerr << endl;
                cerr << name() << "::run:  WARNING: not converged within " << props.maxiter << " passes (" << toc() - tic << " seconds)...final maxdiff:" << maxDiff << endl;
        } else {
            if( props.verbose >= 2 )
                cerr << name() << "::run:  ";
                cerr << "converged in " << _iters << " passes (" << toc() - tic << " seconds)." << endl;
        }
    }

    return maxDiff;
}


} // end of namespace dai
