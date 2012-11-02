/*  This file is part of libDAI - http://www.libdai.org/
 *
 *  Copyright (c) 2006-2011, The libDAI authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
 */


#include <vector>
#include <stack>
#include <dai/daialg.h>


namespace dai {


using namespace std;


Factor calcMarginal( const InfAlg &obj, const VarSet &vs, bool reInit ) {
    Factor Pvs (vs);

    InfAlg *clamped = obj.clone();
    if( !reInit )
        clamped->init();

    map<Var,size_t> varindices;
    for( VarSet::const_iterator n = vs.begin(); n != vs.end(); n++ )
        varindices[*n] = obj.fg().findVar( *n );

    Real logZ0 = -INFINITY;
    for( State s(vs); s.valid(); s++ ) {
        // save unclamped factors connected to vs
        clamped->backupFactors( vs );

        // set clamping Factors to delta functions
        for( VarSet::const_iterator n = vs.begin(); n != vs.end(); n++ )
            clamped->clamp( varindices[*n], s(*n) );

        // run DAIAlg, calc logZ, store in Pvs
        if( reInit )
            clamped->init();
        else
            clamped->init(vs);

        Real logZ;
        try {
            clamped->run();
            logZ = clamped->logZ();
        } catch( Exception &e ) {
            if( e.getCode() == Exception::NOT_NORMALIZABLE )
                logZ = -INFINITY;
            else
                throw;
        }

        if( logZ0 == -INFINITY )
            if( logZ != -INFINITY )
                logZ0 = logZ;

        if( logZ == -INFINITY )
            Pvs.set( s, 0 );
        else
            Pvs.set( s, exp(logZ - logZ0) ); // subtract logZ0 to avoid very large numbers

        // restore clamped factors
        clamped->restoreFactors( vs );
    }

    delete clamped;

    return( Pvs.normalized() );
}


vector<Factor> calcPairBeliefs( const InfAlg & obj, const VarSet& vs, bool reInit, bool accurate ) {
    vector<Factor> result;
    size_t N = vs.size();
    result.reserve( N * (N - 1) / 2 );

    InfAlg *clamped = obj.clone();
    if( !reInit )
        clamped->init();

    map<Var,size_t> varindices;
    for( VarSet::const_iterator v = vs.begin(); v != vs.end(); v++ )
        varindices[*v] = obj.fg().findVar( *v );

    if( accurate ) {
        Real logZ0 = 0.0;
        VarSet::const_iterator nj = vs.begin();
        for( long j = 0; j < (long)N - 1; j++, nj++ ) {
            size_t k = 0;
            for( VarSet::const_iterator nk = nj; (++nk) != vs.end(); k++ ) {
                Factor pairbelief( VarSet(*nj, *nk) );

                // clamp Vars j and k to their possible values
                for( size_t j_val = 0; j_val < nj->states(); j_val++ )
                    for( size_t k_val = 0; k_val < nk->states(); k_val++ ) {
                        // save unclamped factors connected to vs
                        clamped->backupFactors( vs );

                        clamped->clamp( varindices[*nj], j_val );
                        clamped->clamp( varindices[*nk], k_val );
                        if( reInit )
                            clamped->init();
                        else
                            clamped->init(vs);

                        Real logZ;
                        try {
                            clamped->run();
                            logZ = clamped->logZ();
                        } catch( Exception &e ) {
                            if( e.getCode() == Exception::NOT_NORMALIZABLE )
                                logZ = -INFINITY;
                            else
                                throw;
                        }

                        if( logZ0 == -INFINITY )
                            if( logZ != -INFINITY )
                                logZ0 = logZ;

                        Real Z_xj;
                        if( logZ == -INFINITY )
                            Z_xj = 0;
                        else
                            Z_xj = exp(logZ - logZ0); // subtract logZ0 to avoid very large numbers

                        // we assume that j.label() < k.label()
                        // i.e. we make an assumption here about the indexing
                        pairbelief.set( j_val + (k_val * nj->states()), Z_xj );

                        // restore clamped factors
                        clamped->restoreFactors( vs );
                    }

                result.push_back( pairbelief.normalized() );
            }
        }
    } else {
        // convert vs to vector<VarSet>
        vector<Var> vvs( vs.begin(), vs.end() );

        vector<Factor> pairbeliefs;
        pairbeliefs.reserve( N * N );
        for( size_t j = 0; j < N; j++ )
            for( size_t k = 0; k < N; k++ )
                if( j == k )
                    pairbeliefs.push_back( Factor() );
                else
                    pairbeliefs.push_back( Factor( VarSet(vvs[j], vvs[k]) ) );

        Real logZ0 = -INFINITY;
        for( size_t j = 0; j < N; j++ ) {
            // clamp Var j to its possible values
            for( size_t j_val = 0; j_val < vvs[j].states(); j_val++ ) {
                clamped->clamp( varindices[vvs[j]], j_val, true );
                if( reInit )
                    clamped->init();
                else
                    clamped->init(vs);

                Real logZ;
                try {
                    clamped->run();
                    logZ = clamped->logZ();
                } catch( Exception &e ) {
                    if( e.getCode() == Exception::NOT_NORMALIZABLE )
                        logZ = -INFINITY;
                    else
                        throw;
                }

                if( logZ0 == -INFINITY )
                    if( logZ != -INFINITY )
                        logZ0 = logZ;

                Real Z_xj;
                if( logZ == -INFINITY )
                    Z_xj = 0;
                else
                    Z_xj = exp(logZ - logZ0); // subtract logZ0 to avoid very large numbers

                for( size_t k = 0; k < N; k++ )
                    if( k != j ) {
                        Factor b_k = clamped->belief(vvs[k]);
                        for( size_t k_val = 0; k_val < vvs[k].states(); k_val++ )
                            if( vvs[j].label() < vvs[k].label() )
                                pairbeliefs[j * N + k].set( j_val + (k_val * vvs[j].states()), Z_xj * b_k[k_val] );
                            else
                                pairbeliefs[j * N + k].set( k_val + (j_val * vvs[k].states()), Z_xj * b_k[k_val] );
                    }

                // restore clamped factors
                clamped->restoreFactors( vs );
            }
        }

        // Calculate result by taking the geometric average
        for( size_t j = 0; j < N; j++ )
            for( size_t k = j+1; k < N; k++ )
                result.push_back( ((pairbeliefs[j * N + k] * pairbeliefs[k * N + j]) ^ 0.5).normalized() );
    }
    delete clamped;
    return result;
}


std::vector<size_t> findMaximum( const InfAlg& obj ) {
    vector<size_t> maximum( obj.fg().nrVars() );
    vector<bool> visitedVars( obj.fg().nrVars(), false );
    vector<bool> visitedFactors( obj.fg().nrFactors(), false );
    stack<size_t> scheduledFactors;
    scheduledFactors.push( 0 );
    while( !scheduledFactors.empty() ) {
        size_t I = scheduledFactors.top();
        scheduledFactors.pop();
        if( visitedFactors[I] )
            continue;
        visitedFactors[I] = true;

        // Get marginal of factor I
        Prob probF = obj.beliefF(I).p();

        // The allowed configuration is restrained according to the variables assigned so far:
        // pick the argmax amongst the allowed states
        Real maxProb = -numeric_limits<Real>::max();
        State maxState( obj.fg().factor(I).vars() );
        size_t maxcount = 0;
        for( State s( obj.fg().factor(I).vars() ); s.valid(); ++s ) {
            // First, calculate whether this state is consistent with variables that
            // have been assigned already
            bool allowedState = true;
            foreach( const Neighbor &j, obj.fg().nbF(I) )
                if( visitedVars[j.node] && maximum[j.node] != s(obj.fg().var(j.node)) ) {
                    allowedState = false;
                    break;
                }
            // If it is consistent, check if its probability is larger than what we have seen so far
            if( allowedState ) {
                if( probF[s] > maxProb ) {
                    maxState = s;
                    maxProb = probF[s];
                    maxcount = 1;
                } else
                    maxcount++;
            }
        }
        if( maxProb == 0.0 )
            DAI_THROWE(RUNTIME_ERROR,"Failed to decode the MAP state (should try harder using a SAT solver, but that's not implemented yet)");
        DAI_ASSERT( obj.fg().factor(I).p()[maxState] != 0.0 );

        // Decode the argmax
        foreach( const Neighbor &j, obj.fg().nbF(I) ) {
            if( visitedVars[j.node] ) {
                // We have already visited j earlier - hopefully our state is consistent
                if( maximum[j.node] != maxState( obj.fg().var(j.node) ) )
                    DAI_THROWE(RUNTIME_ERROR,"Detected inconsistency while decoding MAP state (should try harder using a SAT solver, but that's not implemented yet)");
            } else {
                // We found a consistent state for variable j
                visitedVars[j.node] = true;
                maximum[j.node] = maxState( obj.fg().var(j.node) );
                foreach( const Neighbor &J, obj.fg().nbV(j) )
                    if( !visitedFactors[J] )
                        scheduledFactors.push(J);
            }
        }
    }
    return maximum;
}


} // end of namespace dai
