/*  This file is part of libDAI - http://www.libdai.org/
 *
 *  Copyright (c) 2006-2011, The libDAI authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
 */


#include <iostream>
#include <stack>
#include <dai/jtree.h>


namespace dai {


using namespace std;


void JTree::setProperties( const PropertySet &opts ) {
    DAI_ASSERT( opts.hasKey("updates") );

    props.updates = opts.getStringAs<Properties::UpdateType>("updates");
    if( opts.hasKey("verbose") )
        props.verbose = opts.getStringAs<size_t>("verbose");
    else
        props.verbose = 0;
    if( opts.hasKey("inference") )
        props.inference = opts.getStringAs<Properties::InfType>("inference");
    else
        props.inference = Properties::InfType::SUMPROD;
    if( opts.hasKey("heuristic") )
        props.heuristic = opts.getStringAs<Properties::HeuristicType>("heuristic");
    else
        props.heuristic = Properties::HeuristicType::MINFILL;
    if( opts.hasKey("maxmem") )
        props.maxmem = opts.getStringAs<size_t>("maxmem");
    else
        props.maxmem = 0;
}


PropertySet JTree::getProperties() const {
    PropertySet opts;
    opts.set( "verbose", props.verbose );
    opts.set( "updates", props.updates );
    opts.set( "inference", props.inference );
    opts.set( "heuristic", props.heuristic );
    opts.set( "maxmem", props.maxmem );
    return opts;
}


string JTree::printProperties() const {
    stringstream s( stringstream::out );
    s << "[";
    s << "verbose=" << props.verbose << ",";
    s << "updates=" << props.updates << ",";
    s << "heuristic=" << props.heuristic << ",";
    s << "inference=" << props.inference << ",";
    s << "maxmem=" << props.maxmem << "]";
    return s.str();
}


JTree::JTree( const FactorGraph &fg, const PropertySet &opts, bool automatic ) : DAIAlgRG(), _mes(), _logZ(), RTree(), Qa(), Qb(), props() {
    setProperties( opts );

    if( automatic ) {
        // Create ClusterGraph which contains maximal factors as clusters
        ClusterGraph _cg( fg, true );
        if( props.verbose >= 3 )
            cerr << "Initial clusters: " << _cg << endl;

        // Use heuristic to guess optimal elimination sequence
        greedyVariableElimination::eliminationCostFunction ec(NULL);
        switch( (size_t)props.heuristic ) {
            case Properties::HeuristicType::MINNEIGHBORS:
                ec = eliminationCost_MinNeighbors;
                break;
            case Properties::HeuristicType::MINWEIGHT:
                ec = eliminationCost_MinWeight;
                break;
            case Properties::HeuristicType::MINFILL:
                ec = eliminationCost_MinFill;
                break;
            case Properties::HeuristicType::WEIGHTEDMINFILL:
                ec = eliminationCost_WeightedMinFill;
                break;
            default:
                DAI_THROW(UNKNOWN_ENUM_VALUE);
        }
        size_t fudge = 6; // this yields a rough estimate of the memory needed (for some reason not yet clearly understood)
        vector<VarSet> ElimVec = _cg.VarElim( greedyVariableElimination( ec ), props.maxmem / (sizeof(Real) * fudge) ).eraseNonMaximal().clusters();
        if( props.verbose >= 3 )
            cerr << "VarElim result: " << ElimVec << endl;

        // Estimate memory needed (rough upper bound)
        BigInt memneeded = 0;
        foreach( const VarSet& cl, ElimVec )
            memneeded += cl.nrStates();
        memneeded *= sizeof(Real) * fudge;
        if( props.verbose >= 1 ) {
            cerr << "Estimate of needed memory: " << memneeded / 1024 << "kB" << endl;
            cerr << "Maximum memory: ";
            if( props.maxmem )
               cerr << props.maxmem / 1024 << "kB" << endl;
            else
               cerr << "unlimited" << endl;
        }
        if( props.maxmem && memneeded > props.maxmem )
            DAI_THROW(OUT_OF_MEMORY);

        // Generate the junction tree corresponding to the elimination sequence
        GenerateJT( fg, ElimVec );
    }
}


void JTree::construct( const FactorGraph &fg, const std::vector<VarSet> &cl, bool verify ) {
    // Copy the factor graph
    FactorGraph::operator=( fg );

    // Construct a weighted graph (each edge is weighted with the cardinality
    // of the intersection of the nodes, where the nodes are the elements of cl).
    WeightedGraph<int> JuncGraph;
    // Start by connecting all clusters with cluster zero, and weight zero,
    // in order to get a connected weighted graph
    for( size_t i = 1; i < cl.size(); i++ )
        JuncGraph[UEdge(i,0)] = 0;
    for( size_t i = 0; i < cl.size(); i++ ) {
        for( size_t j = i + 1; j < cl.size(); j++ ) {
            size_t w = (cl[i] & cl[j]).size();
            if( w )
                JuncGraph[UEdge(i,j)] = w;
        }
    }
    if( props.verbose >= 3 )
        cerr << "Weightedgraph: " << JuncGraph << endl;

    // Construct maximal spanning tree using Prim's algorithm
    RTree = MaxSpanningTree( JuncGraph, true );
    if( props.verbose >= 3 )
        cerr << "Spanning tree: " << RTree << endl;
    DAI_DEBASSERT( RTree.size() == cl.size() - 1 );

    // Construct corresponding region graph

    // Create outer regions
    _ORs.clear();
    _ORs.reserve( cl.size() );
    for( size_t i = 0; i < cl.size(); i++ )
        _ORs.push_back( FRegion( Factor(cl[i], 1.0), 1.0 ) );

    // For each factor, find an outer region that subsumes that factor.
    // Then, multiply the outer region with that factor.
    _fac2OR.clear();
    _fac2OR.resize( nrFactors(), -1U );
    for( size_t I = 0; I < nrFactors(); I++ ) {
        size_t alpha;
        for( alpha = 0; alpha < nrORs(); alpha++ )
            if( OR(alpha).vars() >> factor(I).vars() ) {
                _fac2OR[I] = alpha;
                break;
            }
        if( verify )
            DAI_ASSERT( alpha != nrORs() );
    }
    recomputeORs();

    // Create inner regions and edges
    _IRs.clear();
    _IRs.reserve( RTree.size() );
    vector<Edge> edges;
    edges.reserve( 2 * RTree.size() );
    for( size_t i = 0; i < RTree.size(); i++ ) {
        edges.push_back( Edge( RTree[i].first, nrIRs() ) );
        edges.push_back( Edge( RTree[i].second, nrIRs() ) );
        // inner clusters have counting number -1, except if they are empty
        VarSet intersection = cl[RTree[i].first] & cl[RTree[i].second];
        _IRs.push_back( Region( intersection, intersection.size() ? -1.0 : 0.0 ) );
    }

    // create bipartite graph
    _G.construct( nrORs(), nrIRs(), edges.begin(), edges.end() );

    // Check counting numbers
#ifdef DAI_DEBUG
    checkCountingNumbers();
#endif

    // Create beliefs
    Qa.clear();
    Qa.reserve( nrORs() );
    for( size_t alpha = 0; alpha < nrORs(); alpha++ )
        Qa.push_back( OR(alpha) );

    Qb.clear();
    Qb.reserve( nrIRs() );
    for( size_t beta = 0; beta < nrIRs(); beta++ )
        Qb.push_back( Factor( IR(beta), 1.0 ) );
}


void JTree::GenerateJT( const FactorGraph &fg, const std::vector<VarSet> &cl ) {
    construct( fg, cl, true );

    // Create messages
    _mes.clear();
    _mes.reserve( nrORs() );
    for( size_t alpha = 0; alpha < nrORs(); alpha++ ) {
        _mes.push_back( vector<Factor>() );
        _mes[alpha].reserve( nbOR(alpha).size() );
        foreach( const Neighbor &beta, nbOR(alpha) )
            _mes[alpha].push_back( Factor( IR(beta), 1.0 ) );
    }

    if( props.verbose >= 3 )
        cerr << "Regiongraph generated by JTree::GenerateJT: " << *this << endl;
}


Factor JTree::belief( const VarSet &vs ) const {
    vector<Factor>::const_iterator beta;
    for( beta = Qb.begin(); beta != Qb.end(); beta++ )
        if( beta->vars() >> vs )
            break;
    if( beta != Qb.end() ) {
        if( props.inference == Properties::InfType::SUMPROD )
            return( beta->marginal(vs) );
        else
            return( beta->maxMarginal(vs) );
    } else {
        vector<Factor>::const_iterator alpha;
        for( alpha = Qa.begin(); alpha != Qa.end(); alpha++ )
            if( alpha->vars() >> vs )
                break;
        if( alpha == Qa.end() ) {
            DAI_THROW(BELIEF_NOT_AVAILABLE);
            return Factor();
        } else {
            if( props.inference == Properties::InfType::SUMPROD )
                return( alpha->marginal(vs) );
            else
                return( alpha->maxMarginal(vs) );
        }
    }
}


vector<Factor> JTree::beliefs() const {
    vector<Factor> result;
    for( size_t beta = 0; beta < nrIRs(); beta++ )
        result.push_back( Qb[beta] );
    for( size_t alpha = 0; alpha < nrORs(); alpha++ )
        result.push_back( Qa[alpha] );
    return result;
}


void JTree::runHUGIN() {
    for( size_t alpha = 0; alpha < nrORs(); alpha++ )
        Qa[alpha] = OR(alpha);

    for( size_t beta = 0; beta < nrIRs(); beta++ )
        Qb[beta].fill( 1.0 );

    // CollectEvidence
    _logZ = 0.0;
    for( size_t i = RTree.size(); (i--) != 0; ) {
//      Make outer region RTree[i].first consistent with outer region RTree[i].second
//      IR(i) = seperator OR(RTree[i].first) && OR(RTree[i].second)
        Factor new_Qb;
        if( props.inference == Properties::InfType::SUMPROD )
            new_Qb = Qa[RTree[i].second].marginal( IR( i ), false );
        else
            new_Qb = Qa[RTree[i].second].maxMarginal( IR( i ), false );

        _logZ += log(new_Qb.normalize());
        Qa[RTree[i].first] *= new_Qb / Qb[i];
        Qb[i] = new_Qb;
    }
    if( RTree.empty() )
        _logZ += log(Qa[0].normalize() );
    else
        _logZ += log(Qa[RTree[0].first].normalize());

    // DistributeEvidence
    for( size_t i = 0; i < RTree.size(); i++ ) {
//      Make outer region RTree[i].second consistent with outer region RTree[i].first
//      IR(i) = seperator OR(RTree[i].first) && OR(RTree[i].second)
        Factor new_Qb;
        if( props.inference == Properties::InfType::SUMPROD )
            new_Qb = Qa[RTree[i].first].marginal( IR( i ) );
        else
            new_Qb = Qa[RTree[i].first].maxMarginal( IR( i ) );

        Qa[RTree[i].second] *= new_Qb / Qb[i];
        Qb[i] = new_Qb;
    }

    // Normalize
    for( size_t alpha = 0; alpha < nrORs(); alpha++ )
        Qa[alpha].normalize();
}


void JTree::runShaferShenoy() {
    // First pass
    _logZ = 0.0;
    for( size_t e = nrIRs(); (e--) != 0; ) {
        // send a message from RTree[e].second to RTree[e].first
        // or, actually, from the seperator IR(e) to RTree[e].first

        size_t i = nbIR(e)[1].node; // = RTree[e].second
        size_t j = nbIR(e)[0].node; // = RTree[e].first
        size_t _e = nbIR(e)[0].dual;

        Factor msg = OR(i);
        foreach( const Neighbor &k, nbOR(i) )
            if( k != e )
                msg *= message( i, k.iter );
        if( props.inference == Properties::InfType::SUMPROD )
            message( j, _e ) = msg.marginal( IR(e), false );
        else
            message( j, _e ) = msg.maxMarginal( IR(e), false );
        _logZ += log( message(j,_e).normalize() );
    }

    // Second pass
    for( size_t e = 0; e < nrIRs(); e++ ) {
        size_t i = nbIR(e)[0].node; // = RTree[e].first
        size_t j = nbIR(e)[1].node; // = RTree[e].second
        size_t _e = nbIR(e)[1].dual;

        Factor msg = OR(i);
        foreach( const Neighbor &k, nbOR(i) )
            if( k != e )
                msg *= message( i, k.iter );
        if( props.inference == Properties::InfType::SUMPROD )
            message( j, _e ) = msg.marginal( IR(e) );
        else
            message( j, _e ) = msg.maxMarginal( IR(e) );
    }

    // Calculate beliefs
    for( size_t alpha = 0; alpha < nrORs(); alpha++ ) {
        Factor piet = OR(alpha);
        foreach( const Neighbor &k, nbOR(alpha) )
            piet *= message( alpha, k.iter );
        if( nrIRs() == 0 ) {
            _logZ += log( piet.normalize() );
            Qa[alpha] = piet;
        } else if( alpha == nbIR(0)[0].node /*RTree[0].first*/ ) {
            _logZ += log( piet.normalize() );
            Qa[alpha] = piet;
        } else
            Qa[alpha] = piet.normalized();
    }

    // Only for logZ (and for belief)...
    for( size_t beta = 0; beta < nrIRs(); beta++ ) {
        if( props.inference == Properties::InfType::SUMPROD )
            Qb[beta] = Qa[nbIR(beta)[0].node].marginal( IR(beta) );
        else
            Qb[beta] = Qa[nbIR(beta)[0].node].maxMarginal( IR(beta) );
    }
}


Real JTree::run() {
    if( props.updates == Properties::UpdateType::HUGIN )
        runHUGIN();
    else if( props.updates == Properties::UpdateType::SHSH )
        runShaferShenoy();
    return 0.0;
}


Real JTree::logZ() const {
/*    Real s = 0.0;
    for( size_t beta = 0; beta < nrIRs(); beta++ )
        s += IR(beta).c() * Qb[beta].entropy();
    for( size_t alpha = 0; alpha < nrORs(); alpha++ ) {
        s += OR(alpha).c() * Qa[alpha].entropy();
        s += (OR(alpha).log(true) * Qa[alpha]).sum();
    }
    DAI_ASSERT( abs( _logZ - s ) < 1e-8 );
    return s;*/
    return _logZ;
}


size_t JTree::findEfficientTree( const VarSet& vs, RootedTree &Tree, size_t PreviousRoot ) const {
    // find new root clique (the one with maximal statespace overlap with vs)
    BigInt maxval = 0;
    size_t maxalpha = 0;
    for( size_t alpha = 0; alpha < nrORs(); alpha++ ) {
        BigInt val = VarSet(vs & OR(alpha).vars()).nrStates();
        if( val > maxval ) {
            maxval = val;
            maxalpha = alpha;
        }
    }

    // reorder the tree edges such that maxalpha becomes the new root
    RootedTree newTree( GraphEL( RTree.begin(), RTree.end() ), maxalpha );

    // identify subtree that contains all variables of vs which are not in the new root
    set<DEdge> subTree;
    // for each variable in vs
    for( VarSet::const_iterator n = vs.begin(); n != vs.end(); n++ ) {
        for( size_t e = 0; e < newTree.size(); e++ ) {
            if( OR(newTree[e].second).vars().contains( *n ) ) {
                size_t f = e;
                subTree.insert( newTree[f] );
                size_t pos = newTree[f].first;
                for( ; f > 0; f-- )
                    if( newTree[f-1].second == pos ) {
                        subTree.insert( newTree[f-1] );
                        pos = newTree[f-1].first;
                    }
            }
        }
    }
    if( PreviousRoot != (size_t)-1 && PreviousRoot != maxalpha) {
        // find first occurence of PreviousRoot in the tree, which is closest to the new root
        size_t e = 0;
        for( ; e != newTree.size(); e++ ) {
            if( newTree[e].second == PreviousRoot )
                break;
        }
        DAI_ASSERT( e != newTree.size() );

        // track-back path to root and add edges to subTree
        subTree.insert( newTree[e] );
        size_t pos = newTree[e].first;
        for( ; e > 0; e-- )
            if( newTree[e-1].second == pos ) {
                subTree.insert( newTree[e-1] );
                pos = newTree[e-1].first;
            }
    }

    // Resulting Tree is a reordered copy of newTree
    // First add edges in subTree to Tree
    Tree.clear();
    vector<DEdge> remTree;
    for( RootedTree::const_iterator e = newTree.begin(); e != newTree.end(); e++ )
        if( subTree.count( *e ) )
            Tree.push_back( *e );
        else
            remTree.push_back( *e );
    size_t subTreeSize = Tree.size();
    // Then add remaining edges
    copy( remTree.begin(), remTree.end(), back_inserter( Tree ) );

    return subTreeSize;
}


Factor JTree::calcMarginal( const VarSet& vs ) {
    vector<Factor>::const_iterator beta;
    for( beta = Qb.begin(); beta != Qb.end(); beta++ )
        if( beta->vars() >> vs )
            break;
    if( beta != Qb.end() ) {
        if( props.inference == Properties::InfType::SUMPROD )
            return( beta->marginal(vs) );
        else
            return( beta->maxMarginal(vs) );
    } else {
        vector<Factor>::const_iterator alpha;
        for( alpha = Qa.begin(); alpha != Qa.end(); alpha++ )
            if( alpha->vars() >> vs )
                break;
        if( alpha != Qa.end() ) {
            if( props.inference == Properties::InfType::SUMPROD )
                return( alpha->marginal(vs) );
            else
                return( alpha->maxMarginal(vs) );
        } else {
            // Find subtree to do efficient inference
            RootedTree T;
            size_t Tsize = findEfficientTree( vs, T );

            // Find remaining variables (which are not in the new root)
            VarSet vsrem = vs / OR(T.front().first).vars();
            Factor Pvs (vs, 0.0);

            // Save Qa and Qb on the subtree
            map<size_t,Factor> Qa_old;
            map<size_t,Factor> Qb_old;
            vector<size_t> b(Tsize, 0);
            for( size_t i = Tsize; (i--) != 0; ) {
                size_t alpha1 = T[i].first;
                size_t alpha2 = T[i].second;
                size_t beta;
                for( beta = 0; beta < nrIRs(); beta++ )
                    if( UEdge( RTree[beta].first, RTree[beta].second ) == UEdge( alpha1, alpha2 ) )
                        break;
                DAI_ASSERT( beta != nrIRs() );
                b[i] = beta;

                if( !Qa_old.count( alpha1 ) )
                    Qa_old[alpha1] = Qa[alpha1];
                if( !Qa_old.count( alpha2 ) )
                    Qa_old[alpha2] = Qa[alpha2];
                if( !Qb_old.count( beta ) )
                    Qb_old[beta] = Qb[beta];
            }

            // For all states of vsrem
            for( State s(vsrem); s.valid(); s++ ) {
                // CollectEvidence
                Real logZ = 0.0;
                for( size_t i = Tsize; (i--) != 0; ) {
                // Make outer region T[i].first consistent with outer region T[i].second
                // IR(i) = seperator OR(T[i].first) && OR(T[i].second)

                    for( VarSet::const_iterator n = vsrem.begin(); n != vsrem.end(); n++ )
                        if( Qa[T[i].second].vars() >> *n ) {
                            Factor piet( *n, 0.0 );
                            piet.set( s(*n), 1.0 );
                            Qa[T[i].second] *= piet;
                        }

                    Factor new_Qb;
                    if( props.inference == Properties::InfType::SUMPROD )
                        new_Qb = Qa[T[i].second].marginal( IR( b[i] ), false );
                    else
                        new_Qb = Qa[T[i].second].maxMarginal( IR( b[i] ), false );
                    logZ += log(new_Qb.normalize());
                    Qa[T[i].first] *= new_Qb / Qb[b[i]];
                    Qb[b[i]] = new_Qb;
                }
                logZ += log(Qa[T[0].first].normalize());

                Factor piet( vsrem, 0.0 );
                piet.set( s, exp(logZ) );
                if( props.inference == Properties::InfType::SUMPROD )
                    Pvs += piet * Qa[T[0].first].marginal( vs / vsrem, false );      // OPTIMIZE ME
                else
                    Pvs += piet * Qa[T[0].first].maxMarginal( vs / vsrem, false );      // OPTIMIZE ME

                // Restore clamped beliefs
                for( map<size_t,Factor>::const_iterator alpha = Qa_old.begin(); alpha != Qa_old.end(); alpha++ )
                    Qa[alpha->first] = alpha->second;
                for( map<size_t,Factor>::const_iterator beta = Qb_old.begin(); beta != Qb_old.end(); beta++ )
                    Qb[beta->first] = beta->second;
            }

            return( Pvs.normalized() );
        }
    }
}


std::pair<size_t,BigInt> boundTreewidth( const FactorGraph &fg, greedyVariableElimination::eliminationCostFunction fn, size_t maxStates ) {
    // Create cluster graph from factor graph
    ClusterGraph _cg( fg, true );

    // Obtain elimination sequence
    vector<VarSet> ElimVec = _cg.VarElim( greedyVariableElimination( fn ), maxStates ).eraseNonMaximal().clusters();

    // Calculate treewidth
    size_t treewidth = 0;
    BigInt nrstates = 0.0;
    for( size_t i = 0; i < ElimVec.size(); i++ ) {
        if( ElimVec[i].size() > treewidth )
            treewidth = ElimVec[i].size();
        BigInt s = ElimVec[i].nrStates();
        if( s > nrstates )
            nrstates = s;
    }

    return make_pair(treewidth, nrstates);
}


std::vector<size_t> JTree::findMaximum() const {
    vector<size_t> maximum( nrVars() );
    vector<bool> visitedVars( nrVars(), false );
    vector<bool> visitedORs( nrORs(), false );
    stack<size_t> scheduledORs;
    scheduledORs.push( 0 );
    while( !scheduledORs.empty() ) {
        size_t alpha = scheduledORs.top();
        scheduledORs.pop();
        if( visitedORs[alpha] )
            continue;
        visitedORs[alpha] = true;

        // Get marginal of outer region alpha 
        Prob probF = Qa[alpha].p();

        // The allowed configuration is restrained according to the variables assigned so far:
        // pick the argmax amongst the allowed states
        Real maxProb = -numeric_limits<Real>::max();
        State maxState( OR(alpha).vars() );
        size_t maxcount = 0;
        for( State s( OR(alpha).vars() ); s.valid(); ++s ) {
            // First, calculate whether this state is consistent with variables that
            // have been assigned already
            bool allowedState = true;
            foreach( const Var& j, OR(alpha).vars() ) {
                size_t j_index = findVar(j);
                if( visitedVars[j_index] && maximum[j_index] != s(j) ) {
                    allowedState = false;
                    break;
                }
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
        DAI_ASSERT( maxProb != 0.0 );
        DAI_ASSERT( Qa[alpha][maxState] != 0.0 );

        // Decode the argmax
        foreach( const Var& j, OR(alpha).vars() ) {
            size_t j_index = findVar(j);
            if( visitedVars[j_index] ) {
                // We have already visited j earlier - hopefully our state is consistent
                if( maximum[j_index] != maxState( j ) )
                    DAI_THROWE(RUNTIME_ERROR,"MAP state inconsistent due to loops");
            } else {
                // We found a consistent state for variable j
                visitedVars[j_index] = true;
                maximum[j_index] = maxState( j );
                foreach( const Neighbor &beta, nbOR(alpha) )
                    foreach( const Neighbor &alpha2, nbIR(beta) )
                        if( !visitedORs[alpha2] )
                            scheduledORs.push(alpha2);
            }
        }
    }
    return maximum;
}


} // end of namespace dai
