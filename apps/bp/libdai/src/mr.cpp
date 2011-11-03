/*  This file is part of libDAI - http://www.libdai.org/
 *
 *  Copyright (c) 2006-2011, The libDAI authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
 */


#include <cstdio>
#include <ctime>
#include <cmath>
#include <cstdlib>
#include <dai/mr.h>
#include <dai/bp.h>
#include <dai/jtree.h>
#include <dai/util.h>
#include <dai/bbp.h>


namespace dai {


using namespace std;


void MR::setProperties( const PropertySet &opts ) {
    DAI_ASSERT( opts.hasKey("tol") );
    DAI_ASSERT( opts.hasKey("updates") );
    DAI_ASSERT( opts.hasKey("inits") );

    props.tol = opts.getStringAs<Real>("tol");
    props.updates = opts.getStringAs<Properties::UpdateType>("updates");
    props.inits = opts.getStringAs<Properties::InitType>("inits");
    if( opts.hasKey("verbose") )
        props.verbose = opts.getStringAs<size_t>("verbose");
    else
        props.verbose = 0;
}


PropertySet MR::getProperties() const {
    PropertySet opts;
    opts.set( "tol", props.tol );
    opts.set( "verbose", props.verbose );
    opts.set( "updates", props.updates );
    opts.set( "inits", props.inits );
    return opts;
}


string MR::printProperties() const {
    stringstream s( stringstream::out );
    s << "[";
    s << "tol=" << props.tol << ",";
    s << "verbose=" << props.verbose << ",";
    s << "updates=" << props.updates << ",";
    s << "inits=" << props.inits << "]";
    return s.str();
}


Real MR::T(size_t i, sub_nb A) {
    sub_nb _nbi_min_A(G.nb(i).size());
    _nbi_min_A.set();
    _nbi_min_A &= ~A;

    Real res = theta[i];
    for( size_t _j = 0; _j < _nbi_min_A.size(); _j++ )
        if( _nbi_min_A.test(_j) )
            res += atanh(tJ[i][_j] * M[i][_j]);
    return tanh(res);
}


Real MR::T(size_t i, size_t _j) {
    sub_nb j(G.nb(i).size());
    j.set(_j);
    return T(i,j);
}


Real MR::Omega(size_t i, size_t _j, size_t _l) {
    sub_nb jl(G.nb(i).size());
    jl.set(_j);
    jl.set(_l);
    Real Tijl = T(i,jl);
    return Tijl / (1.0 + tJ[i][_l] * M[i][_l] * Tijl);
}


Real MR::Gamma(size_t i, size_t _j, size_t _l1, size_t _l2) {
    sub_nb jll(G.nb(i).size());
    jll.set(_j);
    Real Tij = T(i,jll);
    jll.set(_l1);
    jll.set(_l2);
    Real Tijll = T(i,jll);

    return (Tijll - Tij) / (1.0 + tJ[i][_l1] * tJ[i][_l2] * M[i][_l1] * M[i][_l2] + tJ[i][_l1] * M[i][_l1] * Tijll + tJ[i][_l2] * M[i][_l2] * Tijll);
}


Real MR::Gamma(size_t i, size_t _l1, size_t _l2) {
    sub_nb ll(G.nb(i).size());
    Real Ti = T(i,ll);
    ll.set(_l1);
    ll.set(_l2);
    Real Till = T(i,ll);

    return (Till - Ti) / (1.0 + tJ[i][_l1] * tJ[i][_l2] * M[i][_l1] * M[i][_l2] + tJ[i][_l1] * M[i][_l1] * Till + tJ[i][_l2] * M[i][_l2] * Till);
}


Real MR::_tJ(size_t i, sub_nb A) {
    sub_nb::size_type _j = A.find_first();
    if( _j == sub_nb::npos )
        return 1.0;
    else
        return tJ[i][_j] * _tJ(i, A.reset(_j));
}


Real MR::appM(size_t i, sub_nb A) {
    sub_nb::size_type _j = A.find_first();
    if( _j == sub_nb::npos )
        return 1.0;
    else {
        sub_nb A_j(A); A_j.reset(_j);

        Real result = M[i][_j] * appM(i, A_j);
        for( size_t _k = 0; _k < A_j.size(); _k++ )
            if( A_j.test(_k) ) {
                sub_nb A_jk(A_j); A_jk.reset(_k);
                result += cors[i][_j][_k] * appM(i,A_jk);
            }

        return result;
    }
}


void MR::sum_subs(size_t j, sub_nb A, Real *sum_even, Real *sum_odd) {
    *sum_even = 0.0;
    *sum_odd = 0.0;

    sub_nb B(A.size());
    do {
        if( B.count() % 2 )
            *sum_odd += _tJ(j,B) * appM(j,B);
        else
            *sum_even += _tJ(j,B) * appM(j,B);

        // calc next subset B
        size_t bit = 0;
        for( ; bit < A.size(); bit++ )
            if( A.test(bit) ) {
                if( B.test(bit) )
                    B.reset(bit);
                else {
                    B.set(bit);
                    break;
                }
            }
    } while (!B.none());
}


void MR::propagateCavityFields() {
    Real sum_even, sum_odd;
    Real maxdev;
    size_t maxruns = 1000;

    for( size_t i = 0; i < G.nrNodes(); i++ )
        foreach( const Neighbor &j, G.nb(i) )
            M[i][j.iter] = 0.1;

    size_t run=0;
    do {
        maxdev=0.0;
        run++;
        for( size_t i = 0; i < G.nrNodes(); i++ ) {
            foreach( const Neighbor &j, G.nb(i) ) {
                size_t _j = j.iter;
                size_t _i = G.findNb(j,i);
                DAI_ASSERT( G.nb(j,_i) == i );

                Real newM = 0.0;
                if( props.updates == Properties::UpdateType::FULL ) {
                    // find indices in nb(j) that do not correspond with i
                    sub_nb _nbj_min_i(G.nb(j).size());
                    _nbj_min_i.set();
                    _nbj_min_i.reset(_i);

                    // find indices in nb(i) that do not correspond with j
                    sub_nb _nbi_min_j(G.nb(i).size());
                    _nbi_min_j.set();
                    _nbi_min_j.reset(_j);

                    sum_subs(j, _nbj_min_i, &sum_even, &sum_odd);
                    newM = (tanh(theta[j]) * sum_even + sum_odd) / (sum_even + tanh(theta[j]) * sum_odd);

                    sum_subs(i, _nbi_min_j, &sum_even, &sum_odd);
                    Real denom = sum_even + tanh(theta[i]) * sum_odd;
                    Real numer = 0.0;
                    for(size_t _k=0; _k < G.nb(i).size(); _k++) if(_k != _j) {
                        sub_nb _nbi_min_jk(_nbi_min_j);
                        _nbi_min_jk.reset(_k);
                        sum_subs(i, _nbi_min_jk, &sum_even, &sum_odd);
                        numer += tJ[i][_k] * cors[i][_j][_k] * (tanh(theta[i]) * sum_even + sum_odd);
                    }
                    newM -= numer / denom;
                } else if( props.updates == Properties::UpdateType::LINEAR ) {
                    newM = T(j,_i);
                    for(size_t _l=0; _l<G.nb(i).size(); _l++) if( _l != _j )
                        newM -= Omega(i,_j,_l) * tJ[i][_l] * cors[i][_j][_l];
                    for(size_t _l1=0; _l1<G.nb(j).size(); _l1++) if( _l1 != _i )
                        for( size_t _l2=_l1+1; _l2<G.nb(j).size(); _l2++) if( _l2 != _i)
                            newM += Gamma(j,_i,_l1,_l2) * tJ[j][_l1] * tJ[j][_l2] * cors[j][_l1][_l2];
                }

                Real dev = newM - M[i][_j];
//              dev *= 0.02;
                if( abs(dev) >= maxdev )
                    maxdev = abs(dev);

                newM = M[i][_j] + dev;
                if( abs(newM) > 1.0 )
                    newM = (newM > 0.0) ? 1.0 : -1.0;
                M[i][_j] = newM;
            }
        }
    } while((maxdev>props.tol)&&(run<maxruns));

    _iters = run;
    if( maxdev > _maxdiff )
        _maxdiff = maxdev;

    if(run==maxruns){
        if( props.verbose >= 1 )
            cerr << "MR::propagateCavityFields: Convergence not reached (maxdev=" << maxdev << ")..." << endl;
    }
}


void MR::calcMagnetizations() {
    for( size_t i = 0; i < G.nrNodes(); i++ ) {
        if( props.updates == Properties::UpdateType::FULL ) {
            // find indices in nb(i)
            sub_nb _nbi( G.nb(i).size() );
            _nbi.set();

            // calc numerator1 and denominator1
            Real sum_even, sum_odd;
            sum_subs(i, _nbi, &sum_even, &sum_odd);

            Mag[i] = (tanh(theta[i]) * sum_even + sum_odd) / (sum_even + tanh(theta[i]) * sum_odd);

        } else if( props.updates == Properties::UpdateType::LINEAR ) {
            sub_nb empty( G.nb(i).size() );
            Mag[i] = T(i,empty);

            for( size_t _l1 = 0; _l1 < G.nb(i).size(); _l1++ )
                for( size_t _l2 = _l1 + 1; _l2 < G.nb(i).size(); _l2++ )
                    Mag[i] += Gamma(i,_l1,_l2) * tJ[i][_l1] * tJ[i][_l2] * cors[i][_l1][_l2];
        }
        if( abs( Mag[i] ) > 1.0 )
            Mag[i] = (Mag[i] > 0.0) ? 1.0 : -1.0;
    }
}


Real MR::calcCavityCorrelations() {
    Real md = 0.0;
    for( size_t i = 0; i < nrVars(); i++ ) {
        vector<Factor> pairq;
        if( props.inits == Properties::InitType::EXACT ) {
            JTree jtcav(*this, PropertySet()("updates", string("HUGIN"))("verbose", (size_t)0) );
            jtcav.makeCavity( i );
            pairq = calcPairBeliefs( jtcav, delta(i), false, true );
        } else if( props.inits == Properties::InitType::CLAMPING ) {
            BP bpcav(*this, PropertySet()("updates", string("SEQMAX"))("tol", (Real)1.0e-9)("maxiter", (size_t)10000)("verbose", (size_t)0)("logdomain", false));
            bpcav.makeCavity( i );

            pairq = calcPairBeliefs( bpcav, delta(i), false, true );
            md = std::max( md, bpcav.maxDiff() );
        } else if( props.inits == Properties::InitType::RESPPROP ) {
            BP bpcav(*this, PropertySet()("updates", string("SEQMAX"))("tol", (Real)1.0e-9)("maxiter", (size_t)10000)("verbose", (size_t)0)("logdomain", false));
            bpcav.makeCavity( i );
            bpcav.makeCavity( i );
            bpcav.init();
            bpcav.run();

            BBP bbp( &bpcav, PropertySet()("verbose",(size_t)0)("tol",(Real)1.0e-9)("maxiter",(size_t)10000)("damping",(Real)0.0)("updates",string("SEQ_MAX")) );
            foreach( const Neighbor &j, G.nb(i) ) {
                // Create weights for magnetization of some spin
                Prob p( 2, 0.0 );
                p.set( 0, -1.0 );
                p.set( 1, 1.0 );

                // BBP cost function would be the magnetization of spin j
                vector<Prob> b1_adj;
                b1_adj.reserve( nrVars() );
                for( size_t l = 0; l < nrVars(); l++ )
                    if( l == j )
                        b1_adj.push_back( p );
                    else
                        b1_adj.push_back( Prob( 2, 0.0 ) );
                bbp.init_V( b1_adj );

                // run BBP to estimate adjoints
                bbp.run();

                foreach( const Neighbor &k, G.nb(i) ) {
                    if( k != j )
                        cors[i][j.iter][k.iter] = (bbp.adj_psi_V(k)[1] - bbp.adj_psi_V(k)[0]);
                    else
                        cors[i][j.iter][k.iter] = 0.0;
                }
            }
        }

        if( props.inits != Properties::InitType::RESPPROP ) {
            for( size_t jk = 0; jk < pairq.size(); jk++ ) {
                VarSet::const_iterator kit = pairq[jk].vars().begin();
                size_t j = findVar( *(kit) );
                size_t k = findVar( *(++kit) );
                pairq[jk].normalize();
                Real cor = (pairq[jk][3] - pairq[jk][2] - pairq[jk][1] + pairq[jk][0]) - (pairq[jk][3] + pairq[jk][2] - pairq[jk][1] - pairq[jk][0]) * (pairq[jk][3] - pairq[jk][2] + pairq[jk][1] - pairq[jk][0]);

                size_t _j = G.findNb(i,j);
                size_t _k = G.findNb(i,k);
                cors[i][_j][_k] = cor;
                cors[i][_k][_j] = cor;
            }
        }

    }
    return md;
}


Real MR::run() {
    if( supported ) {
        if( props.verbose >= 1 )
            cerr << "Starting " << identify() << "...";

        double tic = toc();

        // approximate correlations of cavity spins
        Real md = calcCavityCorrelations();
        if( md > _maxdiff )
            _maxdiff = md;

        // solve messages
        propagateCavityFields();

        // calculate magnetizations
        calcMagnetizations();

        if( props.verbose >= 1 )
            cerr << name() << " needed " << toc() - tic << " seconds." << endl;

        return _maxdiff;
    } else
        return 1.0;
}


Factor MR::beliefV( size_t i ) const {
    if( supported ) {
        Real x[2];
        x[0] = 0.5 - Mag[i] / 2.0;
        x[1] = 0.5 + Mag[i] / 2.0;

        return Factor( var(i), x );
    } else
        return Factor();
}

    
Factor MR::belief (const VarSet &ns) const {
    if( ns.size() == 0 )
        return Factor();
    else if( ns.size() == 1 )
        return beliefV( findVar( *(ns.begin()) ) );
    else {
        DAI_THROW(BELIEF_NOT_AVAILABLE);
        return Factor();
    }
}


vector<Factor> MR::beliefs() const {
    vector<Factor> result;
    for( size_t i = 0; i < nrVars(); i++ )
        result.push_back( beliefV( i ) );
    return result;
}


MR::MR( const FactorGraph &fg, const PropertySet &opts ) : DAIAlgFG(fg), supported(true), _maxdiff(0.0), _iters(0) {
    setProperties( opts );

    size_t N = fg.nrVars();

    // check whether all vars in fg are binary
    for( size_t i = 0; i < N; i++ )
        if( (fg.var(i).states() > 2) ) {
            supported = false;
            break;
        }
    if( !supported )
        DAI_THROWE(NOT_IMPLEMENTED,"MR only supports binary variables");

    // check whether all interactions are pairwise or single
    // and construct Markov graph
    G = GraphAL(N);
    for( size_t I = 0; I < fg.nrFactors(); I++ ) {
        const Factor &psi = fg.factor(I);
        if( psi.vars().size() > 2 ) {
            supported = false;
            break;
        } else if( psi.vars().size() == 2 ) {
            VarSet::const_iterator jit = psi.vars().begin();
            size_t i = fg.findVar( *(jit) );
            size_t j = fg.findVar( *(++jit) );
            G.addEdge( i, j, false );
        }
    }
    if( !supported )
        DAI_THROWE(NOT_IMPLEMENTED,"MR does not support higher order interactions (only single and pairwise are supported)");

    // construct theta
    theta.clear();
    theta.resize( N, 0.0 );

    // construct tJ
    tJ.resize( N );
    for( size_t i = 0; i < N; i++ )
        tJ[i].resize( G.nb(i).size(), 0.0 );

    // initialize theta and tJ
    for( size_t I = 0; I < fg.nrFactors(); I++ ) {
        const Factor &psi = fg.factor(I);
        if( psi.vars().size() == 1 ) {
            size_t i = fg.findVar( *(psi.vars().begin()) );
            theta[i] += 0.5 * log(psi[1] / psi[0]);
        } else if( psi.vars().size() == 2 ) {
            VarSet::const_iterator jit = psi.vars().begin();
            size_t i = fg.findVar( *(jit) );
            size_t j = fg.findVar( *(++jit) );

            Real w_ij = 0.25 * log(psi[3] * psi[0] / (psi[2] * psi[1]));
            tJ[i][G.findNb(i,j)] += w_ij;
            tJ[j][G.findNb(j,i)] += w_ij;

            theta[i] += 0.25 * log(psi[3] / psi[2] * psi[1] / psi[0]);
            theta[j] += 0.25 * log(psi[3] / psi[1] * psi[2] / psi[0]);
        }
    }
    for( size_t i = 0; i < N; i++ )
        foreach( const Neighbor &j, G.nb(i) )
            tJ[i][j.iter] = tanh( tJ[i][j.iter] );

    // construct M
    M.resize( N );
    for( size_t i = 0; i < N; i++ )
        M[i].resize( G.nb(i).size() );

    // construct cors
    cors.resize( N );
    for( size_t i = 0; i < N; i++ )
        cors[i].resize( G.nb(i).size() );
    for( size_t i = 0; i < N; i++ )
        for( size_t _j = 0; _j < cors[i].size(); _j++ )
            cors[i][_j].resize( G.nb(i).size() );

    // construct Mag
    Mag.resize( N );
}


} // end of namespace dai
