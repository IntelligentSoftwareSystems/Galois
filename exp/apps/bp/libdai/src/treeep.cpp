/*  This file is part of libDAI - http://www.libdai.org/
 *
 *  Copyright (c) 2006-2011, The libDAI authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
 */


#include <iostream>
#include <fstream>
#include <vector>
#include <dai/jtree.h>
#include <dai/treeep.h>
#include <dai/util.h>


namespace dai {


using namespace std;


void TreeEP::setProperties( const PropertySet &opts ) {
    DAI_ASSERT( opts.hasKey("tol") );
    DAI_ASSERT( opts.hasKey("type") );

    props.tol = opts.getStringAs<Real>("tol");
    props.type = opts.getStringAs<Properties::TypeType>("type");
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
}


PropertySet TreeEP::getProperties() const {
    PropertySet opts;
    opts.set( "tol", props.tol );
    opts.set( "maxiter", props.maxiter );
    opts.set( "maxtime", props.maxtime );
    opts.set( "verbose", props.verbose );
    opts.set( "type", props.type );
    return opts;
}


string TreeEP::printProperties() const {
    stringstream s( stringstream::out );
    s << "[";
    s << "tol=" << props.tol << ",";
    s << "maxiter=" << props.maxiter << ",";
    s << "maxtime=" << props.maxtime << ",";
    s << "verbose=" << props.verbose << ",";
    s << "type=" << props.type << "]";
    return s.str();
}


TreeEP::TreeEP( const FactorGraph &fg, const PropertySet &opts ) : JTree(fg, opts("updates",string("HUGIN")), false), _maxdiff(0.0), _iters(0), props(), _Q() {
    setProperties( opts );

    if( opts.hasKey("tree") ) {
        construct( fg, opts.getAs<RootedTree>("tree") );
    } else {
        if( props.type == Properties::TypeType::ORG || props.type == Properties::TypeType::ALT ) {
            // ORG: construct weighted graph with as weights a crude estimate of the
            // mutual information between the nodes
            // ALT: construct weighted graph with as weights an upper bound on the
            // effective interaction strength between pairs of nodes

            WeightedGraph<Real> wg;
            // in order to get a connected weighted graph, we start
            // by connecting every variable to the zero'th variable with weight 0
            for( size_t i = 1; i < fg.nrVars(); i++ )
                wg[UEdge(i,0)] = 0.0;
            for( size_t i = 0; i < fg.nrVars(); i++ ) {
                SmallSet<size_t> delta_i = fg.bipGraph().delta1( i, false );
                const Var& v_i = fg.var(i);
                foreach( size_t j, delta_i ) 
                    if( i < j ) {
                        const Var& v_j = fg.var(j);
                        VarSet v_ij( v_i, v_j );
                        SmallSet<size_t> nb_ij = fg.bipGraph().nb1Set( i ) | fg.bipGraph().nb1Set( j );
                        Factor piet;
                        foreach( size_t I, nb_ij ) {
                            const VarSet& Ivars = fg.factor(I).vars();
                            if( props.type == Properties::TypeType::ORG ) {
                                if( (Ivars == v_i) || (Ivars == v_j) )
                                    piet *= fg.factor(I);
                                else if( Ivars >> v_ij )
                                    piet *= fg.factor(I).marginal( v_ij );
                            } else {
                                if( Ivars >> v_ij )
                                    piet *= fg.factor(I);
                            }
                        }
                        if( props.type == Properties::TypeType::ORG ) {
                            if( piet.vars() >> v_ij ) {
                                piet = piet.marginal( v_ij );
                                Factor pietf = piet.marginal(v_i) * piet.marginal(v_j);
                                wg[UEdge(i,j)] = dist( piet, pietf, DISTKL );
                            } else {
                                // this should never happen...
                                DAI_ASSERT( 0 == 1 );
                                wg[UEdge(i,j)] = 0;
                            }
                        } else
                            wg[UEdge(i,j)] = piet.strength(v_i, v_j);
                    }
            }

            // find maximal spanning tree
            if( props.verbose >= 3 )
                cerr << "WeightedGraph: " << wg << endl;
            RootedTree t = MaxSpanningTree( wg, true );
            if( props.verbose >= 3 )
                cerr << "Spanningtree: " << t << endl;
            construct( fg, t );
        } else
            DAI_THROW(UNKNOWN_ENUM_VALUE);
    }
}


void TreeEP::construct( const FactorGraph& fg, const RootedTree& tree ) {
    // Copy the factor graph
    FactorGraph::operator=( fg );

    vector<VarSet> cl;
    for( size_t i = 0; i < tree.size(); i++ )
        cl.push_back( VarSet( var(tree[i].first), var(tree[i].second) ) );

    // If no outer region can be found subsuming that factor, label the
    // factor as off-tree.
    JTree::construct( *this, cl, false );

    if( props.verbose >= 1 )
        cerr << "TreeEP::construct: The tree has size " << JTree::RTree.size() << endl;
    if( props.verbose >= 3 )
        cerr << "  it is " << JTree::RTree << " with cliques " << cl << endl;

    // Create factor approximations
    _Q.clear();
    size_t PreviousRoot = (size_t)-1;
    // Second repetition: previous root of first off-tree factor should be the root of the last off-tree factor
    for( size_t repeats = 0; repeats < 2; repeats++ )
        for( size_t I = 0; I < nrFactors(); I++ )
            if( offtree(I) ) {
                // find efficient subtree
                RootedTree subTree;
                size_t subTreeSize = findEfficientTree( factor(I).vars(), subTree, PreviousRoot );
                PreviousRoot = subTree[0].first;
                subTree.resize( subTreeSize );
                if( props.verbose >= 1 )
                    cerr << "Subtree " << I << " has size " << subTreeSize << endl;
                if( props.verbose >= 3 )
                    cerr << "  it is " << subTree << endl;
                _Q[I] = TreeEPSubTree( subTree, RTree, Qa, Qb, &factor(I) );
                if( repeats == 1 )
                    break;
            }

    if( props.verbose >= 3 )
        cerr << "Resulting regiongraph: " << *this << endl;
}


void TreeEP::init() {
    runHUGIN();

    // Init factor approximations
    for( size_t I = 0; I < nrFactors(); I++ )
        if( offtree(I) )
            _Q[I].init();
}


Real TreeEP::run() {
    if( props.verbose >= 1 )
        cerr << "Starting " << identify() << "...";
    if( props.verbose >= 3 )
        cerr << endl;

    double tic = toc();

    vector<Factor> oldBeliefs = beliefs();

    // do several passes over the network until maximum number of iterations has
    // been reached or until the maximum belief difference is smaller than tolerance
    Real maxDiff = INFINITY;
    for( _iters = 0; _iters < props.maxiter && maxDiff > props.tol && (toc() - tic) < props.maxtime; _iters++ ) {
        for( size_t I = 0; I < nrFactors(); I++ )
            if( offtree(I) ) {
                _Q[I].InvertAndMultiply( Qa, Qb );
                _Q[I].HUGIN_with_I( Qa, Qb );
                _Q[I].InvertAndMultiply( Qa, Qb );
            }

        // calculate new beliefs and compare with old ones
        vector<Factor> newBeliefs = beliefs();
        maxDiff = -INFINITY;
        for( size_t t = 0; t < oldBeliefs.size(); t++ )
            maxDiff = std::max( maxDiff, dist( newBeliefs[t], oldBeliefs[t], DISTLINF ) );
        swap( newBeliefs, oldBeliefs );

        if( props.verbose >= 3 )
            cerr << name() << "::run:  maxdiff " << maxDiff << " after " << _iters+1 << " passes" << endl;
    }

    if( maxDiff > _maxdiff )
        _maxdiff = maxDiff;

    if( props.verbose >= 1 ) {
        if( maxDiff > props.tol ) {
            if( props.verbose == 1 )
                cerr << endl;
            cerr << name() << "::run:  WARNING: not converged after " << _iters << " passes (" << toc() - tic << " seconds)...final maxdiff:" << maxDiff << endl;
        } else {
            if( props.verbose >= 3 )
                cerr << name() << "::run:  ";
            cerr << "converged in " << _iters << " passes (" << toc() - tic << " seconds)." << endl;
        }
    }

    return maxDiff;
}


Real TreeEP::logZ() const {
    Real s = 0.0;

    // entropy of the tree
    for( size_t beta = 0; beta < nrIRs(); beta++ )
        s -= Qb[beta].entropy();
    for( size_t alpha = 0; alpha < nrORs(); alpha++ )
        s += Qa[alpha].entropy();

    // energy of the on-tree factors
    for( size_t alpha = 0; alpha < nrORs(); alpha++ )
        s += (OR(alpha).log(true) * Qa[alpha]).sum();

    // energy of the off-tree factors
    for( size_t I = 0; I < nrFactors(); I++ )
        if( offtree(I) )
            s += (_Q.find(I))->second.logZ( Qa, Qb );

    return s;
}


TreeEP::TreeEPSubTree::TreeEPSubTree( const RootedTree &subRTree, const RootedTree &jt_RTree, const std::vector<Factor> &jt_Qa, const std::vector<Factor> &jt_Qb, const Factor *I ) : _Qa(), _Qb(), _RTree(), _a(), _b(), _I(I), _ns(), _nsrem(), _logZ(0.0) {
    _ns = _I->vars();

    // Make _Qa, _Qb, _a and _b corresponding to the subtree
    _b.reserve( subRTree.size() );
    _Qb.reserve( subRTree.size() );
    _RTree.reserve( subRTree.size() );
    for( size_t i = 0; i < subRTree.size(); i++ ) {
        size_t alpha1 = subRTree[i].first;  // old index 1
        size_t alpha2 = subRTree[i].second; // old index 2
        size_t beta;                        // old sep index
        for( beta = 0; beta < jt_RTree.size(); beta++ )
            if( UEdge( jt_RTree[beta].first, jt_RTree[beta].second ) == UEdge( alpha1, alpha2 ) )
                break;
        DAI_ASSERT( beta != jt_RTree.size() );

        size_t newalpha1 = find(_a.begin(), _a.end(), alpha1) - _a.begin();
        if( newalpha1 == _a.size() ) {
            _Qa.push_back( Factor( jt_Qa[alpha1].vars(), 1.0 ) );
            _a.push_back( alpha1 );         // save old index in index conversion table
        }

        size_t newalpha2 = find(_a.begin(), _a.end(), alpha2) - _a.begin();
        if( newalpha2 == _a.size() ) {
            _Qa.push_back( Factor( jt_Qa[alpha2].vars(), 1.0 ) );
            _a.push_back( alpha2 );         // save old index in index conversion table
        }

        _RTree.push_back( DEdge( newalpha1, newalpha2 ) );
        _Qb.push_back( Factor( jt_Qb[beta].vars(), 1.0 ) );
        _b.push_back( beta );
    }

    // Find remaining variables (which are not in the new root)
    _nsrem = _ns / _Qa[0].vars();
}


void TreeEP::TreeEPSubTree::init() {
    for( size_t alpha = 0; alpha < _Qa.size(); alpha++ )
        _Qa[alpha].fill( 1.0 );
    for( size_t beta = 0; beta < _Qb.size(); beta++ )
        _Qb[beta].fill( 1.0 );
}


void TreeEP::TreeEPSubTree::InvertAndMultiply( const std::vector<Factor> &Qa, const std::vector<Factor> &Qb ) {
    for( size_t alpha = 0; alpha < _Qa.size(); alpha++ )
        _Qa[alpha] = Qa[_a[alpha]] / _Qa[alpha];

    for( size_t beta = 0; beta < _Qb.size(); beta++ )
        _Qb[beta] = Qb[_b[beta]] / _Qb[beta];
}


void TreeEP::TreeEPSubTree::HUGIN_with_I( std::vector<Factor> &Qa, std::vector<Factor> &Qb ) {
    // Backup _Qa and _Qb
    vector<Factor> _Qa_old(_Qa);
    vector<Factor> _Qb_old(_Qb);

    // Clear Qa and Qb
    for( size_t alpha = 0; alpha < _Qa.size(); alpha++ )
        Qa[_a[alpha]].fill( 0.0 );
    for( size_t beta = 0; beta < _Qb.size(); beta++ )
        Qb[_b[beta]].fill( 0.0 );

    // For all states of _nsrem
    for( State s(_nsrem); s.valid(); s++ ) {
        // Multiply root with slice of I
        _Qa[0] *= _I->slice( _nsrem, s );

        // CollectEvidence
        for( size_t i = _RTree.size(); (i--) != 0; ) {
            // clamp variables in nsrem
            for( VarSet::const_iterator n = _nsrem.begin(); n != _nsrem.end(); n++ )
                if( _Qa[_RTree[i].second].vars() >> *n )
                    _Qa[_RTree[i].second] *= createFactorDelta( *n, s(*n) );
            Factor new_Qb = _Qa[_RTree[i].second].marginal( _Qb[i].vars(), false );
            _Qa[_RTree[i].first] *= new_Qb / _Qb[i];
            _Qb[i] = new_Qb;
        }

        // DistributeEvidence
        for( size_t i = 0; i < _RTree.size(); i++ ) {
            Factor new_Qb = _Qa[_RTree[i].first].marginal( _Qb[i].vars(), false );
            _Qa[_RTree[i].second] *= new_Qb / _Qb[i];
            _Qb[i] = new_Qb;
        }

        // Store Qa's and Qb's
        for( size_t alpha = 0; alpha < _Qa.size(); alpha++ )
            Qa[_a[alpha]].p() += _Qa[alpha].p();
        for( size_t beta = 0; beta < _Qb.size(); beta++ )
            Qb[_b[beta]].p() += _Qb[beta].p();

        // Restore _Qa and _Qb
        _Qa = _Qa_old;
        _Qb = _Qb_old;
    }

    // Normalize Qa and Qb
    _logZ = 0.0;
    for( size_t alpha = 0; alpha < _Qa.size(); alpha++ ) {
        _logZ += log(Qa[_a[alpha]].sum());
        Qa[_a[alpha]].normalize();
    }
    for( size_t beta = 0; beta < _Qb.size(); beta++ ) {
        _logZ -= log(Qb[_b[beta]].sum());
        Qb[_b[beta]].normalize();
    }
}


Real TreeEP::TreeEPSubTree::logZ( const std::vector<Factor> &Qa, const std::vector<Factor> &Qb ) const {
    Real s = 0.0;
    for( size_t alpha = 0; alpha < _Qa.size(); alpha++ )
        s += (Qa[_a[alpha]] * _Qa[alpha].log(true)).sum();
    for( size_t beta = 0; beta < _Qb.size(); beta++ )
        s -= (Qb[_b[beta]] * _Qb[beta].log(true)).sum();
    return s + _logZ;
}


} // end of namespace dai
