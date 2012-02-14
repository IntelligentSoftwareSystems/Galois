/*  This file is part of libDAI - http://www.libdai.org/
 *
 *  Copyright (c) 2006-2011, The libDAI authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
 */


#include <iostream>
#include <iomanip>
#include <iterator>
#include <map>
#include <set>
#include <fstream>
#include <string>
#include <algorithm>
#include <functional>
#include <dai/factorgraph.h>
#include <dai/util.h>
#include <dai/exceptions.h>
#include <boost/lexical_cast.hpp>


namespace dai {


using namespace std;


FactorGraph::FactorGraph( const std::vector<Factor> &P ) : _G(), _backup() {
    // add factors, obtain variables
    set<Var> varset;
    _factors.reserve( P.size() );
    size_t nrEdges = 0;
    for( vector<Factor>::const_iterator p2 = P.begin(); p2 != P.end(); p2++ ) {
        _factors.push_back( *p2 );
        copy( p2->vars().begin(), p2->vars().end(), inserter( varset, varset.begin() ) );
        nrEdges += p2->vars().size();
    }

    // add vars
    _vars.reserve( varset.size() );
    for( set<Var>::const_iterator p1 = varset.begin(); p1 != varset.end(); p1++ )
        _vars.push_back( *p1 );

    // create graph structure
    constructGraph( nrEdges );
}


void FactorGraph::constructGraph( size_t nrEdges ) {
    // create a mapping for indices
    hash_map<size_t, size_t> hashmap;

    for( size_t i = 0; i < vars().size(); i++ )
        hashmap[var(i).label()] = i;

    // create edge list
    vector<Edge> edges;
    edges.reserve( nrEdges );
    for( size_t i2 = 0; i2 < nrFactors(); i2++ ) {
        const VarSet& ns = factor(i2).vars();
        for( VarSet::const_iterator q = ns.begin(); q != ns.end(); q++ )
            edges.push_back( Edge(hashmap[q->label()], i2) );
    }

    // create bipartite graph
    _G.construct( nrVars(), nrFactors(), edges.begin(), edges.end() );
}


/// Writes a FactorGraph to an output stream
std::ostream& operator<< ( std::ostream &os, const FactorGraph &fg ) {
    os << fg.nrFactors() << endl;

    for( size_t I = 0; I < fg.nrFactors(); I++ ) {
        os << endl;
        os << fg.factor(I).vars().size() << endl;
        for( VarSet::const_iterator i = fg.factor(I).vars().begin(); i != fg.factor(I).vars().end(); i++ )
            os << i->label() << " ";
        os << endl;
        for( VarSet::const_iterator i = fg.factor(I).vars().begin(); i != fg.factor(I).vars().end(); i++ )
            os << i->states() << " ";
        os << endl;
        size_t nr_nonzeros = 0;
        for( size_t k = 0; k < fg.factor(I).nrStates(); k++ )
            if( fg.factor(I)[k] != (Real)0 )
                nr_nonzeros++;
        os << nr_nonzeros << endl;
        for( size_t k = 0; k < fg.factor(I).nrStates(); k++ )
            if( fg.factor(I)[k] != (Real)0 )
                os << k << " " << setw(os.precision()+4) << fg.factor(I)[k] << endl;
    }

    return(os);
}


/// Reads a FactorGraph from an input stream
std::istream& operator>> ( std::istream& is, FactorGraph &fg ) {
    long verbose = 0;

    vector<Factor> facs;
    size_t nr_Factors;
    string line;

    while( (is.peek()) == '#' )
        getline(is,line);
    is >> nr_Factors;
    if( is.fail() )
        DAI_THROWE(INVALID_FACTORGRAPH_FILE,"Cannot read number of factors");
    if( verbose >= 1 )
        cerr << "Reading " << nr_Factors << " factors..." << endl;

    getline (is,line);
    if( is.fail() || line.size() > 0 )
        DAI_THROWE(INVALID_FACTORGRAPH_FILE,"Expecting empty line");

    map<long,size_t> vardims;
    for( size_t I = 0; I < nr_Factors; I++ ) {
        if( verbose >= 2 )
            cerr << "Reading factor " << I << "..." << endl;
        size_t nr_members;
        while( (is.peek()) == '#' )
            getline(is,line);
        is >> nr_members;
        if( verbose >= 2 )
            cerr << "  nr_members: " << nr_members << endl;

        vector<long> labels;
        for( size_t mi = 0; mi < nr_members; mi++ ) {
            long mi_label;
            while( (is.peek()) == '#' )
                getline(is,line);
            is >> mi_label;
            labels.push_back(mi_label);
        }
        if( verbose >= 2 )
            cerr << "  labels: " << labels << endl;

        vector<size_t> dims;
        for( size_t mi = 0; mi < nr_members; mi++ ) {
            size_t mi_dim;
            while( (is.peek()) == '#' )
                getline(is,line);
            is >> mi_dim;
            dims.push_back(mi_dim);
        }
        if( verbose >= 2 )
            cerr << "  dimensions: " << dims << endl;

        // add the Factor
        vector<Var> Ivars;
        Ivars.reserve( nr_members );
        for( size_t mi = 0; mi < nr_members; mi++ ) {
            map<long,size_t>::iterator vdi = vardims.find( labels[mi] );
            if( vdi != vardims.end() ) {
                // check whether dimensions are consistent
                if( vdi->second != dims[mi] )
                    DAI_THROWE(INVALID_FACTORGRAPH_FILE,"Variable with label " + boost::lexical_cast<string>(labels[mi]) + " has inconsistent dimensions.");
            } else
                vardims[labels[mi]] = dims[mi];
            Ivars.push_back( Var(labels[mi], dims[mi]) );
        }
        facs.push_back( Factor( VarSet( Ivars.begin(), Ivars.end(), Ivars.size() ), (Real)0 ) );
        if( verbose >= 2 )
            cerr << "  vardims: " << vardims << endl;

        // calculate permutation object
        Permute permindex( Ivars );

        // read values
        size_t nr_nonzeros;
        while( (is.peek()) == '#' )
            getline(is,line);
        is >> nr_nonzeros;
        if( verbose >= 2 )
            cerr << "  nonzeroes: " << nr_nonzeros << endl;
        for( size_t k = 0; k < nr_nonzeros; k++ ) {
            size_t li;
            Real val;
            while( (is.peek()) == '#' )
                getline(is,line);
            is >> li;
            while( (is.peek()) == '#' )
                getline(is,line);
            is >> val;

            // store value, but permute indices first according to internal representation
            facs.back().set( permindex.convertLinearIndex( li ), val );
        }
    }

    if( verbose >= 3 )
        cerr << "factors:" << facs << endl;

    fg = FactorGraph(facs);

    return is;
}


VarSet FactorGraph::Delta( size_t i ) const {
    // calculate Markov Blanket
    VarSet Del;
    foreach( const Neighbor &I, nbV(i) ) // for all neighboring factors I of i
        foreach( const Neighbor &j, nbF(I) ) // for all neighboring variables j of I
            Del |= var(j);

    return Del;
}


VarSet FactorGraph::Delta( const VarSet &ns ) const {
    VarSet result;
    for( VarSet::const_iterator n = ns.begin(); n != ns.end(); n++ )
        result |= Delta( findVar(*n) );
    return result;
}


void FactorGraph::makeCavity( size_t i, bool backup ) {
    // fills all Factors that include var(i) with ones
    map<size_t,Factor> newFacs;
    foreach( const Neighbor &I, nbV(i) ) // for all neighboring factors I of i
        newFacs[I] = Factor( factor(I).vars(), (Real)1 );
    setFactors( newFacs, backup );
}


void FactorGraph::ReadFromFile( const char *filename ) {
    ifstream infile;
    infile.open( filename );
    if( infile.is_open() ) {
        infile >> *this;
        infile.close();
    } else
        DAI_THROWE(CANNOT_READ_FILE,"Cannot read from file " + std::string(filename));
}


void FactorGraph::WriteToFile( const char *filename, size_t precision ) const {
    ofstream outfile;
    outfile.open( filename );
    if( outfile.is_open() ) {
        outfile.precision( precision );
        outfile << *this;
        outfile.close();
    } else
        DAI_THROWE(CANNOT_WRITE_FILE,"Cannot write to file " + std::string(filename));
}


void FactorGraph::printDot( std::ostream &os ) const {
    os << "graph FactorGraph {" << endl;
    os << "node[shape=circle,width=0.4,fixedsize=true];" << endl;
    for( size_t i = 0; i < nrVars(); i++ )
        os << "\tv" << var(i).label() << ";" << endl;
    os << "node[shape=box,width=0.3,height=0.3,fixedsize=true];" << endl;
    for( size_t I = 0; I < nrFactors(); I++ )
        os << "\tf" << I << ";" << endl;
    for( size_t i = 0; i < nrVars(); i++ )
        foreach( const Neighbor &I, nbV(i) )  // for all neighboring factors I of i
            os << "\tv" << var(i).label() << " -- f" << I << ";" << endl;
    os << "}" << endl;
}


GraphAL FactorGraph::MarkovGraph() const {
    GraphAL G( nrVars() );
    for( size_t i = 0; i < nrVars(); i++ )
        foreach( const Neighbor &I, nbV(i) )
            foreach( const Neighbor &j, nbF(I) )
                if( i < j )
                    G.addEdge( i, j, true );
    return G;
}


bool FactorGraph::isMaximal( size_t I ) const {
    const VarSet& I_vars = factor(I).vars();
    size_t I_size = I_vars.size();

    if( I_size == 0 ) {
        for( size_t J = 0; J < nrFactors(); J++ ) 
            if( J != I )
                if( factor(J).vars().size() > 0 )
                    return false;
        return true;
    } else {
        foreach( const Neighbor& i, nbF(I) ) {
            foreach( const Neighbor& J, nbV(i) ) {
                if( J != I )
                    if( (factor(J).vars() >> I_vars) && (factor(J).vars().size() != I_size) )
                        return false;
            }
        }
        return true;
    }
}


size_t FactorGraph::maximalFactor( size_t I ) const {
    const VarSet& I_vars = factor(I).vars();
    size_t I_size = I_vars.size();

    if( I_size == 0 ) {
        for( size_t J = 0; J < nrFactors(); J++ )
            if( J != I )
                if( factor(J).vars().size() > 0 )
                    return maximalFactor( J );
        return I;
    } else {
        foreach( const Neighbor& i, nbF(I) ) {
            foreach( const Neighbor& J, nbV(i) ) {
                if( J != I )
                    if( (factor(J).vars() >> I_vars) && (factor(J).vars().size() != I_size) )
                        return maximalFactor( J );
            }
        }
        return I;
    }
}


vector<VarSet> FactorGraph::maximalFactorDomains() const {
    vector<VarSet> result;

    for( size_t I = 0; I < nrFactors(); I++ )
        if( isMaximal( I ) )
            result.push_back( factor(I).vars() );

    if( result.size() == 0 )
        result.push_back( VarSet() );
    return result;
}


Real FactorGraph::logScore( const std::vector<size_t>& statevec ) const {
    // Construct a State object that represents statevec
    // This decouples the representation of the joint state in statevec from the factor graph
    map<Var, size_t> statemap;
    for( size_t i = 0; i < statevec.size(); i++ )
        statemap[var(i)] = statevec[i];
    State S(statemap);

    // Evaluate the log probability of the joint configuration in statevec
    // by summing the log factor entries of the factors that correspond to this joint configuration
    Real lS = 0.0;
    for( size_t I = 0; I < nrFactors(); I++ )
        lS += dai::log( factor(I)[BigInt_size_t(S(factor(I).vars()))] );
    return lS;
}


void FactorGraph::clamp( size_t i, size_t x, bool backup ) {
    DAI_ASSERT( x <= var(i).states() );
    Factor mask( var(i), (Real)0 );
    mask.set( x, (Real)1 );

    map<size_t, Factor> newFacs;
    foreach( const Neighbor &I, nbV(i) )
        newFacs[I] = factor(I) * mask;
    setFactors( newFacs, backup );

    return;
}


void FactorGraph::clampVar( size_t i, const vector<size_t> &is, bool backup ) {
    Var n = var(i);
    Factor mask_n( n, (Real)0 );

    foreach( size_t i, is ) {
        DAI_ASSERT( i <= n.states() );
        mask_n.set( i, (Real)1 );
    }

    map<size_t, Factor> newFacs;
    foreach( const Neighbor &I, nbV(i) )
        newFacs[I] = factor(I) * mask_n;
    setFactors( newFacs, backup );
}


void FactorGraph::clampFactor( size_t I, const vector<size_t> &is, bool backup ) {
    size_t st = factor(I).nrStates();
    Factor newF( factor(I).vars(), (Real)0 );

    foreach( size_t i, is ) {
        DAI_ASSERT( i <= st );
        newF.set( i, factor(I)[i] );
    }

    setFactor( I, newF, backup );
}


void FactorGraph::backupFactor( size_t I ) {
    map<size_t,Factor>::iterator it = _backup.find( I );
    if( it != _backup.end() )
        DAI_THROW(MULTIPLE_UNDO);
    _backup[I] = factor(I);
}


void FactorGraph::restoreFactor( size_t I ) {
    map<size_t,Factor>::iterator it = _backup.find( I );
    if( it != _backup.end() ) {
        setFactor(I, it->second);
        _backup.erase(it);
    } else
        DAI_THROW(OBJECT_NOT_FOUND);
}


void FactorGraph::backupFactors( const VarSet &ns ) {
    for( size_t I = 0; I < nrFactors(); I++ )
        if( factor(I).vars().intersects( ns ) )
            backupFactor( I );
}


void FactorGraph::restoreFactors( const VarSet &ns ) {
    map<size_t,Factor> facs;
    for( map<size_t,Factor>::iterator uI = _backup.begin(); uI != _backup.end(); ) {
        if( factor(uI->first).vars().intersects( ns ) ) {
            facs.insert( *uI );
            _backup.erase(uI++);
        } else
            uI++;
    }
    setFactors( facs );
}


void FactorGraph::restoreFactors() {
    setFactors( _backup );
    _backup.clear();
}


void FactorGraph::backupFactors( const std::set<size_t> & facs ) {
    for( std::set<size_t>::const_iterator fac = facs.begin(); fac != facs.end(); fac++ )
        backupFactor( *fac );
}


bool FactorGraph::isPairwise() const {
    bool pairwise = true;
    for( size_t I = 0; I < nrFactors() && pairwise; I++ )
        if( factor(I).vars().size() > 2 )
            pairwise = false;
    return pairwise;
}


bool FactorGraph::isBinary() const {
    bool binary = true;
    for( size_t i = 0; i < nrVars() && binary; i++ )
        if( var(i).states() > 2 )
            binary = false;
    return binary;
}


FactorGraph FactorGraph::clamped( size_t i, size_t state ) const {
    Var v = var( i );
    Real zeroth_order = (Real)1;
    vector<Factor> clamped_facs;
    clamped_facs.push_back( createFactorDelta( v, state ) );
    for( size_t I = 0; I < nrFactors(); I++ ) {
        VarSet v_I = factor(I).vars();
        Factor new_factor;
        if( v_I.intersects( v ) )
            new_factor = factor(I).slice( v, state );
        else
            new_factor = factor(I);

        if( new_factor.vars().size() != 0 ) {
            size_t J = 0;
            // if it can be merged with a previous one, do that
            for( J = 0; J < clamped_facs.size(); J++ )
                if( clamped_facs[J].vars() == new_factor.vars() ) {
                    clamped_facs[J] *= new_factor;
                    break;
                }
            // otherwise, push it back
            if( J == clamped_facs.size() || clamped_facs.size() == 0 )
                clamped_facs.push_back( new_factor );
        } else
            zeroth_order *= new_factor[0];
    }
    *(clamped_facs.begin()) *= zeroth_order;
    return FactorGraph( clamped_facs );
}


FactorGraph FactorGraph::maximalFactors() const {
    vector<size_t> maxfac( nrFactors() );
    map<size_t,size_t> newindex;
    size_t nrmax = 0;
    for( size_t I = 0; I < nrFactors(); I++ ) {
        maxfac[I] = I;
        VarSet maxfacvars = factor(maxfac[I]).vars();
        for( size_t J = 0; J < nrFactors(); J++ ) {
            VarSet Jvars = factor(J).vars();
            if( Jvars >> maxfacvars && (Jvars != maxfacvars) ) {
                maxfac[I] = J;
                maxfacvars = factor(maxfac[I]).vars();
            }
        }
        if( maxfac[I] == I )
            newindex[I] = nrmax++;
    }

    vector<Factor> facs( nrmax );
    for( size_t I = 0; I < nrFactors(); I++ )
        facs[newindex[maxfac[I]]] *= factor(I);

    return FactorGraph( facs.begin(), facs.end(), vars().begin(), vars().end(), facs.size(), nrVars() );
}


} // end of namespace dai
