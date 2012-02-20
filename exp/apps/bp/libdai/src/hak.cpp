/*  This file is part of libDAI - http://www.libdai.org/
 *
 *  Copyright (c) 2006-2011, The libDAI authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
 */


#include <map>
#include <dai/hak.h>
#include <dai/util.h>
#include <dai/exceptions.h>


namespace dai {


using namespace std;


/// Sets factor entries that lie between 0 and \a epsilon to \a epsilon
template <class T>
TFactor<T>& makePositive( TFactor<T> &f, T epsilon ) {
    for( size_t t = 0; t < f.states(); t++ )
        if( (0 < f[t]) && (f[t] < epsilon) )
            f[t] = epsilon;
    return f;
}

/// Sets factor entries that are smaller (in absolute value) than \a epsilon to 0
template <class T>
TFactor<T>& makeZero( TFactor<T> &f, T epsilon ) {
    for( size_t t = 0; t < f.states(); t++ )
        if( f[t] < epsilon && f[t] > -epsilon )
            f[t] = 0;
    return f;
}


void HAK::setProperties( const PropertySet &opts ) {
    DAI_ASSERT( opts.hasKey("tol") );
    DAI_ASSERT( opts.hasKey("doubleloop") );
    DAI_ASSERT( opts.hasKey("clusters") );

    props.tol = opts.getStringAs<Real>("tol");
    props.doubleloop = opts.getStringAs<bool>("doubleloop");
    props.clusters = opts.getStringAs<Properties::ClustersType>("clusters");

    if( opts.hasKey("maxiter") )
        props.maxiter = opts.getStringAs<size_t>("maxiter");
    else
        props.maxiter = 10000;
    if( opts.hasKey("maxtime") )
        props.maxtime = opts.getStringAs<Real>("maxtime");
    else
        props.maxtime = INFINITY;
    if( opts.hasKey("verbose") )
        props.verbose = opts.getStringAs<size_t>("verbose");
    else
        props.verbose = 0;
    if( opts.hasKey("loopdepth") )
        props.loopdepth = opts.getStringAs<size_t>("loopdepth");
    else
        DAI_ASSERT( props.clusters != Properties::ClustersType::LOOP );
    if( opts.hasKey("damping") )
        props.damping = opts.getStringAs<Real>("damping");
    else
        props.damping = 0.0;
    if( opts.hasKey("init") )
        props.init = opts.getStringAs<Properties::InitType>("init");
    else
        props.init = Properties::InitType::UNIFORM;
}


PropertySet HAK::getProperties() const {
    PropertySet opts;
    opts.set( "tol", props.tol );
    opts.set( "maxiter", props.maxiter );
    opts.set( "maxtime", props.maxtime );
    opts.set( "verbose", props.verbose );
    opts.set( "doubleloop", props.doubleloop );
    opts.set( "clusters", props.clusters );
    opts.set( "init", props.init );
    opts.set( "loopdepth", props.loopdepth );
    opts.set( "damping", props.damping );
    return opts;
}


string HAK::printProperties() const {
    stringstream s( stringstream::out );
    s << "[";
    s << "tol=" << props.tol << ",";
    s << "maxiter=" << props.maxiter << ",";
    s << "maxtime=" << props.maxtime << ",";
    s << "verbose=" << props.verbose << ",";
    s << "doubleloop=" << props.doubleloop << ",";
    s << "clusters=" << props.clusters << ",";
    s << "init=" << props.init << ",";
    s << "loopdepth=" << props.loopdepth << ",";
    s << "damping=" << props.damping << "]";
    return s.str();
}


void HAK::construct() {
    // Create outer beliefs
    if( props.verbose >= 3 )
        cerr << "Constructing outer beliefs" << endl;
    _Qa.clear();
    _Qa.reserve(nrORs());
    for( size_t alpha = 0; alpha < nrORs(); alpha++ )
        _Qa.push_back( Factor( OR(alpha) ) );

    // Create inner beliefs
    if( props.verbose >= 3 )
        cerr << "Constructing inner beliefs" << endl;
    _Qb.clear();
    _Qb.reserve(nrIRs());
    for( size_t beta = 0; beta < nrIRs(); beta++ )
        _Qb.push_back( Factor( IR(beta) ) );

    // Create messages
    if( props.verbose >= 3 )
        cerr << "Constructing messages" << endl;
    _muab.clear();
    _muab.reserve( nrORs() );
    _muba.clear();
    _muba.reserve( nrORs() );
    for( size_t alpha = 0; alpha < nrORs(); alpha++ ) {
        _muab.push_back( vector<Factor>() );
        _muba.push_back( vector<Factor>() );
        _muab[alpha].reserve( nbOR(alpha).size() );
        _muba[alpha].reserve( nbOR(alpha).size() );
        foreach( const Neighbor &beta, nbOR(alpha) ) {
            _muab[alpha].push_back( Factor( IR(beta) ) );
            _muba[alpha].push_back( Factor( IR(beta) ) );
        }
    }
}


HAK::HAK( const RegionGraph &rg, const PropertySet &opts ) : DAIAlgRG(rg), _Qa(), _Qb(), _muab(), _muba(), _maxdiff(0.0), _iters(0U), props() {
    setProperties( opts );

    construct();
}


void HAK::findLoopClusters( const FactorGraph & fg, std::set<VarSet> &allcl, VarSet newcl, const Var & root, size_t length, VarSet vars ) {
    for( VarSet::const_iterator in = vars.begin(); in != vars.end(); in++ ) {
        VarSet ind = fg.delta( fg.findVar( *in ) );
        if( (newcl.size()) >= 2 && ind.contains( root ) )
            allcl.insert( newcl | *in );
        else if( length > 1 )
            findLoopClusters( fg, allcl, newcl | *in, root, length - 1, ind / newcl );
    }
}


HAK::HAK(const FactorGraph & fg, const PropertySet &opts) : DAIAlgRG(), _Qa(), _Qb(), _muab(), _muba(), _maxdiff(0.0), _iters(0U), props() {
    setProperties( opts );

    if( props.verbose >= 3 )
        cerr << "Constructing clusters" << endl;

    vector<VarSet> cl;
    if( props.clusters == Properties::ClustersType::MIN ) {
        cl = fg.maximalFactorDomains();
        constructCVM( fg, cl );
    } else if( props.clusters == Properties::ClustersType::DELTA ) {
        cl.reserve( fg.nrVars() );
        for( size_t i = 0; i < fg.nrVars(); i++ )
            cl.push_back( fg.Delta(i) );
        constructCVM( fg, cl );
    } else if( props.clusters == Properties::ClustersType::LOOP ) {
        cl = fg.maximalFactorDomains();
        set<VarSet> scl;
        if( props.verbose >= 2 )
            cerr << "Searching loops...";
        for( size_t i0 = 0; i0 < fg.nrVars(); i0++ ) {
            VarSet i0d = fg.delta(i0);
            if( props.loopdepth > 1 )
                findLoopClusters( fg, scl, fg.var(i0), fg.var(i0), props.loopdepth - 1, fg.delta(i0) );
        }
        if( props.verbose >= 2 )
            cerr << "done" << endl;
        for( set<VarSet>::const_iterator c = scl.begin(); c != scl.end(); c++ )
            cl.push_back(*c);
        if( props.verbose >= 3 ) {
            cerr << name() << " uses the following clusters: " << endl;
            for( vector<VarSet>::const_iterator cli = cl.begin(); cli != cl.end(); cli++ )
                cerr << *cli << endl;
        }
        constructCVM( fg, cl );
    } else if( props.clusters == Properties::ClustersType::BETHE ) {
        // Copy factor graph structure
        if( props.verbose >= 3 )
            cerr << "Copying factor graph" << endl;
        FactorGraph::operator=( fg );

        // Construct inner regions (single variables)
        if( props.verbose >= 3 )
            cerr << "Constructing inner regions" << endl;
        _IRs.reserve( fg.nrVars() );
        for( size_t i = 0; i < fg.nrVars(); i++ )
            _IRs.push_back( Region( fg.var(i), 1.0 ) );

        // Construct graph
        if( props.verbose >= 3 )
            cerr << "Constructing graph" << endl;
        _G = BipartiteGraph( 0, nrIRs() );

        // Construct outer regions:
        // maximal factors become new outer regions
        // non-maximal factors are assigned an outer region that contains them
        if( props.verbose >= 3 )
            cerr << "Construct outer regions" << endl;
        _fac2OR.reserve( nrFactors() );
        queue<pair<size_t, size_t> > todo;
        for( size_t I = 0; I < fg.nrFactors(); I++ ) {
            size_t J = fg.maximalFactor( I );
            if( J == I ) {
                // I is maximal; add it to the outer regions
                _fac2OR.push_back( nrORs() );
                // Construct outer region (with counting number 1.0)
                _ORs.push_back( FRegion( fg.factor(I), 1.0 ) );
                // Add node and edges to graph
                SmallSet<size_t> irs = fg.bipGraph().nb2Set( I );
                _G.addNode1( irs.begin(), irs.end(), irs.size() );
            } else if( J < I ) {
                // J is larger and has already been assigned to an outer region
                // so I should belong to the same outer region as J
                _fac2OR.push_back( _fac2OR[J] );
                _ORs[_fac2OR[J]] *= fg.factor(I);
            } else {
                // J is larger but has not yet been assigned to an outer region
                // we handle this case later
                _fac2OR.push_back( -1 );
                todo.push( make_pair( I, J ) );
            }
        }
        // finish the construction
        while( !todo.empty() ) {
            size_t I = todo.front().first;
            size_t J = todo.front().second;
            todo.pop();
            _fac2OR[I] = _fac2OR[J];
            _ORs[_fac2OR[J]] *= fg.factor(I);
        }

        // Calculate inner regions' counting numbers
        for( size_t beta = 0; beta < nrIRs(); beta++ )
            _IRs[beta].c() = 1.0 - _G.nb2(beta).size();
    } else
        DAI_THROW(UNKNOWN_ENUM_VALUE);

    construct();

    if( props.verbose >= 3 )
        cerr << name() << " regiongraph: " << *this << endl;
}


void HAK::init( const VarSet &ns ) {
    for( size_t alpha = 0; alpha < nrORs(); alpha++ )
        if( _Qa[alpha].vars().intersects( ns ) ) {
            if( props.init == Properties::InitType::UNIFORM )
                _Qa[alpha].setUniform();
            else
                _Qa[alpha].randomize();
            _Qa[alpha] *= OR(alpha);
            _Qa[alpha].normalize();
        }

    for( size_t beta = 0; beta < nrIRs(); beta++ )
        if( IR(beta).intersects( ns ) ) {
            if( props.init == Properties::InitType::UNIFORM )
                _Qb[beta].fill( 1.0 );
            else
                _Qb[beta].randomize();
            foreach( const Neighbor &alpha, nbIR(beta) ) {
                size_t _beta = alpha.dual;
                if( props.init == Properties::InitType::UNIFORM ) {
                    muab( alpha, _beta ).fill( 1.0 );
                    muba( alpha, _beta ).fill( 1.0 );
                } else {
                    muab( alpha, _beta ).randomize();
                    muba( alpha, _beta ).randomize();
                }
            }
        }
}


void HAK::init() {
    for( size_t alpha = 0; alpha < nrORs(); alpha++ ) {
        if( props.init == Properties::InitType::UNIFORM )
            _Qa[alpha].setUniform();
        else
            _Qa[alpha].randomize();
        _Qa[alpha] *= OR(alpha);
        _Qa[alpha].normalize();
    }

    for( size_t beta = 0; beta < nrIRs(); beta++ )
        if( props.init == Properties::InitType::UNIFORM )
            _Qb[beta].setUniform();
        else
            _Qb[beta].randomize();

    for( size_t alpha = 0; alpha < nrORs(); alpha++ )
        foreach( const Neighbor &beta, nbOR(alpha) ) {
            size_t _beta = beta.iter;
            if( props.init == Properties::InitType::UNIFORM ) {
                muab( alpha, _beta ).setUniform();
                muba( alpha, _beta ).setUniform();
            } else {
                muab( alpha, _beta ).randomize();
                muba( alpha, _beta ).randomize();
            }
        }
}


Real HAK::doGBP() {
    if( props.verbose >= 1 )
        cerr << "Starting " << identify() << "...";
    if( props.verbose >= 3)
        cerr << endl;

    double tic = toc();

    // Check whether counting numbers won't lead to problems
    for( size_t beta = 0; beta < nrIRs(); beta++ )
        DAI_ASSERT( nbIR(beta).size() + IR(beta).c() != 0.0 );

    // Keep old beliefs to check convergence
    vector<Factor> oldBeliefsV;
    oldBeliefsV.reserve( nrVars() );
    for( size_t i = 0; i < nrVars(); i++ )
        oldBeliefsV.push_back( beliefV(i) );
    vector<Factor> oldBeliefsF;
    oldBeliefsF.reserve( nrFactors() );
    for( size_t I = 0; I < nrFactors(); I++ )
        oldBeliefsF.push_back( beliefF(I) );

    // do several passes over the network until maximum number of iterations has
    // been reached or until the maximum belief difference is smaller than tolerance
    Real maxDiff = INFINITY;
    for( _iters = 0; _iters < props.maxiter && maxDiff > props.tol; _iters++ ) {
        for( size_t beta = 0; beta < nrIRs(); beta++ ) {
            foreach( const Neighbor &alpha, nbIR(beta) ) {
                size_t _beta = alpha.dual;
                muab( alpha, _beta ) = _Qa[alpha].marginal(IR(beta)) / muba(alpha,_beta);
                /* TODO: INVESTIGATE THIS PROBLEM
                 *
                 * In some cases, the muab's can have very large entries because the muba's have very
                 * small entries. This may cause NANs later on (e.g., multiplying large quantities may
                 * result in +inf; normalization then tries to calculate inf / inf which is NAN).
                 * A fix of this problem would consist in normalizing the messages muab.
                 * However, it is not obvious whether this is a real solution, because it has a
                 * negative performance impact and the NAN's seem to be a symptom of a fundamental
                 * numerical unstability.
                 */
                 muab(alpha,_beta).normalize();
            }

            Factor Qb_new;
            foreach( const Neighbor &alpha, nbIR(beta) ) {
                size_t _beta = alpha.dual;
                Qb_new *= muab(alpha,_beta) ^ (1 / (nbIR(beta).size() + IR(beta).c()));
            }

            Qb_new.normalize();
            if( Qb_new.hasNaNs() ) {
                // TODO: WHAT TO DO IN THIS CASE?
                cerr << name() << "::doGBP:  Qb_new has NaNs!" << endl;
                return 1.0;
            }
            /* TODO: WHAT IS THE PURPOSE OF THE FOLLOWING CODE?
             *
             *   _Qb[beta] = Qb_new.makeZero(1e-100);
             */

            if( props.doubleloop || props.damping == 0.0 )
                _Qb[beta] = Qb_new; // no damping for double loop
            else
                _Qb[beta] = (Qb_new^(1.0 - props.damping)) * (_Qb[beta]^props.damping);

            foreach( const Neighbor &alpha, nbIR(beta) ) {
                size_t _beta = alpha.dual;
                muba(alpha,_beta) = _Qb[beta] / muab(alpha,_beta);

                /* TODO: INVESTIGATE WHETHER THIS HACK (INVENTED BY KEES) TO PREVENT NANS MAKES SENSE
                 *
                 *   muba(beta,*alpha).makePositive(1e-100);
                 *
                 */

                Factor Qa_new = OR(alpha);
                foreach( const Neighbor &gamma, nbOR(alpha) )
                    Qa_new *= muba(alpha,gamma.iter);
                Qa_new ^= (1.0 / OR(alpha).c());
                Qa_new.normalize();
                if( Qa_new.hasNaNs() ) {
                    cerr << name() << "::doGBP:  Qa_new has NaNs!" << endl;
                    return 1.0;
                }
                /* TODO: WHAT IS THE PURPOSE OF THE FOLLOWING CODE?
                 *
                 *   _Qb[beta] = Qb_new.makeZero(1e-100);
                 */

                if( props.doubleloop || props.damping == 0.0 )
                    _Qa[alpha] = Qa_new; // no damping for double loop
                else
                    // FIXME: GEOMETRIC DAMPING IS SLOW!
                _Qa[alpha] = (Qa_new^(1.0 - props.damping)) * (_Qa[alpha]^props.damping);
            }
        }

        // Calculate new single variable beliefs and compare with old ones
        maxDiff = -INFINITY;
        for( size_t i = 0; i < nrVars(); ++i ) {
            Factor b = beliefV(i);
            maxDiff = std::max( maxDiff, dist( b, oldBeliefsV[i], DISTLINF ) );
            oldBeliefsV[i] = b;
        }
        for( size_t I = 0; I < nrFactors(); ++I ) {
            Factor b = beliefF(I);
            maxDiff = std::max( maxDiff, dist( b, oldBeliefsF[I], DISTLINF ) );
            oldBeliefsF[I] = b;
        }

        if( props.verbose >= 3 )
            cerr << name() << "::doGBP:  maxdiff " << maxDiff << " after " << _iters+1 << " passes" << endl;
    }

    if( maxDiff > _maxdiff )
        _maxdiff = maxDiff;

    if( props.verbose >= 1 ) {
        if( maxDiff > props.tol ) {
            if( props.verbose == 1 )
                cerr << endl;
            cerr << name() << "::doGBP:  WARNING: not converged within " << props.maxiter << " passes (" << toc() - tic << " seconds)...final maxdiff:" << maxDiff << endl;
        } else {
            if( props.verbose >= 2 )
                cerr << name() << "::doGBP:  ";
            cerr << "converged in " << _iters << " passes (" << toc() - tic << " seconds)." << endl;
        }
    }

    return maxDiff;
}


Real HAK::doDoubleLoop() {
    if( props.verbose >= 1 )
        cerr << "Starting " << identify() << "...";
    if( props.verbose >= 3)
        cerr << endl;

    double tic = toc();

    // Save original outer regions
    vector<FRegion> org_ORs = _ORs;

    // Save original inner counting numbers and set negative counting numbers to zero
    vector<Real> org_IR_cs( nrIRs(), 0.0 );
    for( size_t beta = 0; beta < nrIRs(); beta++ ) {
        org_IR_cs[beta] = IR(beta).c();
        if( IR(beta).c() < 0.0 )
            IR(beta).c() = 0.0;
    }

    // Keep old beliefs to check convergence
    vector<Factor> oldBeliefsV;
    oldBeliefsV.reserve( nrVars() );
    for( size_t i = 0; i < nrVars(); i++ )
        oldBeliefsV.push_back( beliefV(i) );
    vector<Factor> oldBeliefsF;
    oldBeliefsF.reserve( nrFactors() );
    for( size_t I = 0; I < nrFactors(); I++ )
        oldBeliefsF.push_back( beliefF(I) );

    size_t outer_maxiter   = props.maxiter;
    Real   outer_tol       = props.tol;
    size_t outer_verbose   = props.verbose;
    Real   org_maxdiff     = _maxdiff;

    // Set parameters for inner loop
    props.maxiter = 5;
    props.verbose = outer_verbose ? outer_verbose - 1 : 0;

    size_t outer_iter = 0;
    size_t total_iter = 0;
    Real maxDiff = INFINITY;
    for( outer_iter = 0; outer_iter < outer_maxiter && maxDiff > outer_tol && (toc() - tic) < props.maxtime; outer_iter++ ) {
        // Calculate new outer regions
        for( size_t alpha = 0; alpha < nrORs(); alpha++ ) {
            OR(alpha) = org_ORs[alpha];
            foreach( const Neighbor &beta, nbOR(alpha) )
                OR(alpha) *= _Qb[beta] ^ ((IR(beta).c() - org_IR_cs[beta]) / nbIR(beta).size());
        }

        // Inner loop
        if( isnan( doGBP() ) )
            return 1.0;

        // Calculate new single variable beliefs and compare with old ones
        maxDiff = -INFINITY;
        for( size_t i = 0; i < nrVars(); ++i ) {
            Factor b = beliefV(i);
            maxDiff = std::max( maxDiff, dist( b, oldBeliefsV[i], DISTLINF ) );
            oldBeliefsV[i] = b;
        }
        for( size_t I = 0; I < nrFactors(); ++I ) {
            Factor b = beliefF(I);
            maxDiff = std::max( maxDiff, dist( b, oldBeliefsF[I], DISTLINF ) );
            oldBeliefsF[I] = b;
        }

        total_iter += Iterations();

        if( props.verbose >= 3 )
            cerr << name() << "::doDoubleLoop:  maxdiff " << maxDiff << " after " << total_iter << " passes" << endl;
    }

    // restore _maxiter, _verbose and _maxdiff
    props.maxiter = outer_maxiter;
    props.verbose = outer_verbose;
    _maxdiff = org_maxdiff;

    _iters = total_iter;
    if( maxDiff > _maxdiff )
        _maxdiff = maxDiff;

    // Restore original outer regions
    _ORs = org_ORs;

    // Restore original inner counting numbers
    for( size_t beta = 0; beta < nrIRs(); ++beta )
        IR(beta).c() = org_IR_cs[beta];

    if( props.verbose >= 1 ) {
        if( maxDiff > props.tol ) {
            if( props.verbose == 1 )
                cerr << endl;
                cerr << name() << "::doDoubleLoop:  WARNING: not converged after " << total_iter << " passes (" << toc() - tic << " seconds)...final maxdiff:" << maxDiff << endl;
            } else {
                if( props.verbose >= 3 )
                    cerr << name() << "::doDoubleLoop:  ";
                cerr << "converged in " << total_iter << " passes (" << toc() - tic << " seconds)." << endl;
            }
        }

    return maxDiff;
}


Real HAK::run() {
    if( props.doubleloop )
        return doDoubleLoop();
    else
        return doGBP();
}


Factor HAK::belief( const VarSet &ns ) const {
    vector<Factor>::const_iterator beta;
    for( beta = _Qb.begin(); beta != _Qb.end(); beta++ )
        if( beta->vars() >> ns )
            break;
    if( beta != _Qb.end() )
        return( beta->marginal(ns) );
    else {
        vector<Factor>::const_iterator alpha;
        for( alpha = _Qa.begin(); alpha != _Qa.end(); alpha++ )
            if( alpha->vars() >> ns )
                break;
        if( alpha == _Qa.end() )
            DAI_THROW(BELIEF_NOT_AVAILABLE);
        return( alpha->marginal(ns) );
    }
}


vector<Factor> HAK::beliefs() const {
    vector<Factor> result;
    for( size_t beta = 0; beta < nrIRs(); beta++ )
        result.push_back( Qb(beta) );
    for( size_t alpha = 0; alpha < nrORs(); alpha++ )
        result.push_back( Qa(alpha) );
    return result;
}


Real HAK::logZ() const {
    Real s = 0.0;
    for( size_t beta = 0; beta < nrIRs(); beta++ )
        s += IR(beta).c() * Qb(beta).entropy();
    for( size_t alpha = 0; alpha < nrORs(); alpha++ ) {
        s += OR(alpha).c() * Qa(alpha).entropy();
        s += (OR(alpha).log(true) * Qa(alpha)).sum();
    }
    return s;
}


} // end of namespace dai
