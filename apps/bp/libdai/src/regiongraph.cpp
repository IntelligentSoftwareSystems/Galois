/*  This file is part of libDAI - http://www.libdai.org/
 *
 *  Copyright (c) 2006-2011, The libDAI authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
 */


#include <algorithm>
#include <cmath>
#include <boost/dynamic_bitset.hpp>
#include <dai/regiongraph.h>
#include <dai/factorgraph.h>
#include <dai/clustergraph.h>


namespace dai {


using namespace std;


void RegionGraph::construct( const FactorGraph &fg, const std::vector<VarSet> &ors, const std::vector<Region> &irs, const std::vector<std::pair<size_t,size_t> > &edges ) {
    // Copy factor graph structure
    FactorGraph::operator=( fg );

    // Copy inner regions
    _IRs = irs;

    // Construct outer regions (giving them counting number 1.0)
    _ORs.clear();
    _ORs.reserve( ors.size() );
    foreach( const VarSet &alpha, ors )
        _ORs.push_back( FRegion(Factor(alpha, 1.0), 1.0) );

    // For each factor, find an outer region that subsumes that factor.
    // Then, multiply the outer region with that factor.
    _fac2OR.clear();
    _fac2OR.reserve( nrFactors() );
    for( size_t I = 0; I < nrFactors(); I++ ) {
        size_t alpha;
        for( alpha = 0; alpha < nrORs(); alpha++ )
            if( OR(alpha).vars() >> factor(I).vars() ) {
                _fac2OR.push_back( alpha );
                break;
            }
        DAI_ASSERT( alpha != nrORs() );
    }
    recomputeORs();

    // Create bipartite graph
    _G.construct( nrORs(), nrIRs(), edges.begin(), edges.end() );
}


void RegionGraph::constructCVM( const FactorGraph &fg, const std::vector<VarSet> &cl, size_t verbose ) {
    if( verbose )
        cerr << "constructCVM called (" << fg.nrVars() << " vars, " << fg.nrFactors() << " facs, " << cl.size() << " clusters)" << endl;

    // Retain only maximal clusters
    if( verbose )
        cerr << "  Constructing ClusterGraph" << endl;
    ClusterGraph cg( cl );
    if( verbose )
        cerr << "  Erasing non-maximal clusters" << endl;
    cg.eraseNonMaximal();

    // Create inner regions - first pass
    if( verbose )
        cerr << "  Creating inner regions (first pass)" << endl;
    set<VarSet> betas;
    for( size_t alpha = 0; alpha < cg.nrClusters(); alpha++ )
        for( size_t alpha2 = alpha; (++alpha2) != cg.nrClusters(); ) {
            VarSet intersection = cg.cluster(alpha) & cg.cluster(alpha2);
            if( intersection.size() > 0 )
                betas.insert( intersection );
        }

    // Create inner regions - subsequent passes
    if( verbose )
        cerr << "  Creating inner regions (next passes)" << endl;
    set<VarSet> new_betas;
    do {
        new_betas.clear();
        for( set<VarSet>::const_iterator gamma = betas.begin(); gamma != betas.end(); gamma++ )
            for( set<VarSet>::const_iterator gamma2 = gamma; (++gamma2) != betas.end(); ) {
                VarSet intersection = (*gamma) & (*gamma2);
                if( (intersection.size() > 0) && (betas.count(intersection) == 0) )
                    new_betas.insert( intersection );
            }
        betas.insert(new_betas.begin(), new_betas.end());
    } while( new_betas.size() );

    // Create inner regions - final phase
    if( verbose )
        cerr << "  Creating inner regions (final phase)" << endl;
    vector<Region> irs;
    irs.reserve( betas.size() );
    for( set<VarSet>::const_iterator beta = betas.begin(); beta != betas.end(); beta++ )
        irs.push_back( Region(*beta,0.0) );

    // Create edges
    if( verbose )
        cerr << "  Creating edges" << endl;
    vector<pair<size_t,size_t> > edges;
    for( size_t beta = 0; beta < irs.size(); beta++ )
        for( size_t alpha = 0; alpha < cg.nrClusters(); alpha++ )
            if( cg.cluster(alpha) >> irs[beta] )
                edges.push_back( pair<size_t,size_t>(alpha,beta) );

    // Construct region graph
    if( verbose )
        cerr << "  Constructing region graph" << endl;
    construct( fg, cg.clusters(), irs, edges );

    // Calculate counting numbers
    if( verbose )
        cerr << "  Calculating counting numbers" << endl;
    calcCVMCountingNumbers();
    
    if( verbose )
        cerr << "Done." << endl;
}


void RegionGraph::calcCVMCountingNumbers() {
    // Calculates counting numbers of inner regions based upon counting numbers of outer regions

    vector<vector<size_t> > ancestors(nrIRs());
    boost::dynamic_bitset<> assigned(nrIRs());
    for( size_t beta = 0; beta < nrIRs(); beta++ ) {
        IR(beta).c() = 0.0;
        for( size_t beta2 = 0; beta2 < nrIRs(); beta2++ )
            if( (beta2 != beta) && IR(beta2) >> IR(beta) )
                ancestors[beta].push_back(beta2);
    }

    bool new_counting;
    do {
        new_counting = false;
        for( size_t beta = 0; beta < nrIRs(); beta++ ) {
            if( !assigned[beta] ) {
                bool has_unassigned_ancestor = false;
                for( vector<size_t>::const_iterator beta2 = ancestors[beta].begin(); (beta2 != ancestors[beta].end()) && !has_unassigned_ancestor; beta2++ )
                    if( !assigned[*beta2] )
                        has_unassigned_ancestor = true;
                if( !has_unassigned_ancestor ) {
                    Real c = 1.0;
                    foreach( const Neighbor &alpha, nbIR(beta) )
                        c -= OR(alpha).c();
                    for( vector<size_t>::const_iterator beta2 = ancestors[beta].begin(); beta2 != ancestors[beta].end(); beta2++ )
                        c -= IR(*beta2).c();
                    IR(beta).c() = c;
                    assigned.set(beta, true);
                    new_counting = true;
                }
            }
        }
    } while( new_counting );
}


bool RegionGraph::checkCountingNumbers() const {
    // Checks whether the counting numbers satisfy the fundamental relation

    bool all_valid = true;
    for( vector<Var>::const_iterator n = vars().begin(); n != vars().end(); n++ ) {
        Real c_n = 0.0;
        for( size_t alpha = 0; alpha < nrORs(); alpha++ )
            if( OR(alpha).vars().contains( *n ) )
                c_n += OR(alpha).c();
        for( size_t beta = 0; beta < nrIRs(); beta++ )
            if( IR(beta).contains( *n ) )
                c_n += IR(beta).c();
        if( fabs(c_n - 1.0) > 1e-15 ) {
            all_valid = false;
            cerr << "WARNING: counting numbers do not satisfy relation for " << *n << "(c_n = " << c_n << ")." << endl;
        }
    }

    return all_valid;
}


void RegionGraph::recomputeORs() {
    for( size_t alpha = 0; alpha < nrORs(); alpha++ )
        OR(alpha).fill( 1.0 );
    for( size_t I = 0; I < nrFactors(); I++ )
        if( fac2OR(I) != -1U )
            OR( fac2OR(I) ) *= factor( I );
}


void RegionGraph::recomputeORs( const VarSet &ns ) {
    for( size_t alpha = 0; alpha < nrORs(); alpha++ )
        if( OR(alpha).vars().intersects( ns ) )
            OR(alpha).fill( 1.0 );
    for( size_t I = 0; I < nrFactors(); I++ )
        if( fac2OR(I) != -1U )
            if( OR( fac2OR(I) ).vars().intersects( ns ) )
                OR( fac2OR(I) ) *= factor( I );
}


void RegionGraph::recomputeOR( size_t I ) {
    DAI_ASSERT( I < nrFactors() );
    if( fac2OR(I) != -1U ) {
        size_t alpha = fac2OR(I);
        OR(alpha).fill( 1.0 );
        for( size_t J = 0; J < nrFactors(); J++ )
            if( fac2OR(J) == alpha )
                OR(alpha) *= factor( J );
    }
}


/// Send RegionGraph to output stream
ostream & operator << (ostream & os, const RegionGraph & rg) {
    os << "digraph RegionGraph {" << endl;
    os << "node[shape=box];" << endl;
    for( size_t alpha = 0; alpha < rg.nrORs(); alpha++ )
        os << "\ta" << alpha << " [label=\"a" << alpha << ": " << rg.OR(alpha).vars() << ", c=" << rg.OR(alpha).c() << "\"];" << endl;
    os << "node[shape=ellipse];" << endl;
    for( size_t beta = 0; beta < rg.nrIRs(); beta++ )
        os << "\tb" << beta << " [label=\"b" << beta << ": " << (VarSet)rg.IR(beta) << ", c=" << rg.IR(beta).c() << "\"];" << endl;
    for( size_t alpha = 0; alpha < rg.nrORs(); alpha++ )
        foreach( const Neighbor &beta, rg.nbOR(alpha) )
            os << "\ta" << alpha << " -> b" << beta << ";" << endl;
    os << "}" << endl;
    return os;
}


} // end of namespace dai
