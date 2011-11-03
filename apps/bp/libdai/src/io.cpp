/*  This file is part of libDAI - http://www.libdai.org/
 *
 *  Copyright (c) 2006-2011, The libDAI authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
 */


#include <dai/io.h>
#include <dai/alldai.h>
#include <iostream>
#include <fstream>


namespace dai {


using namespace std;


void ReadUaiAieFactorGraphFile( const char *filename, size_t verbose, std::vector<Var>& vars, std::vector<Factor>& factors, std::vector<Permute>& permutations ) {
    vars.clear();
    factors.clear();
    permutations.clear();

    // open file
    ifstream is;
    is.open( filename );
    if( is.is_open() ) {
        size_t nrFacs, nrVars;
        string line;
        
        // read header line
        getline(is,line);
        if( is.fail() || line.size() == 0 )
            DAI_THROWE(INVALID_FACTORGRAPH_FILE,"UAI factor graph file should start with nonempty header line");
        if( line[line.size() - 1] == '\r' )
            line.resize( line.size() - 1 ); // for DOS text files
        if( line != "BAYES" && line != "MARKOV" )
            DAI_THROWE(INVALID_FACTORGRAPH_FILE,"UAI factor graph file should start with \"BAYES\" or \"MARKOV\"");
        if( verbose >= 2 )
            cout << "Reading " << line << " network..." << endl;

        // read number of variables
        is >> nrVars;
        if( is.fail() )
            DAI_THROWE(INVALID_FACTORGRAPH_FILE,"Cannot read number of variables");
        if( verbose >= 2 )
            cout << "Reading " << nrVars << " variables..." << endl;

        // for each variable, read its number of states
        vars.reserve( nrVars );
        for( size_t i = 0; i < nrVars; i++ ) {
            size_t dim;
            is >> dim;
            if( is.fail() )
                DAI_THROWE(INVALID_FACTORGRAPH_FILE,"Cannot read number of states of " + toString(i) + "'th variable");
            vars.push_back( Var( i, dim ) );
        }

        // read number of factors
        is >> nrFacs;
        if( is.fail() )
            DAI_THROWE(INVALID_FACTORGRAPH_FILE,"Cannot read number of factors");
        if( verbose >= 2 )
            cout << "Reading " << nrFacs << " factors..." << endl;

        // for each factor, read the variables on which it depends
        vector<vector<Var> > factorVars;
        factors.reserve( nrFacs );
        factorVars.reserve( nrFacs );
        for( size_t I = 0; I < nrFacs; I++ ) {
            if( verbose >= 3 )
                cout << "Reading factor " << I << "..." << endl;

            // read number of variables for factor I
            size_t I_nrVars;
            is >> I_nrVars;
            if( is.fail() )
                DAI_THROWE(INVALID_FACTORGRAPH_FILE,"Cannot read number of variables for " + toString(I) + "'th factor");
            if( verbose >= 3 )
                cout << "  which depends on " << I_nrVars << " variables" << endl;

            // read the variable labels
            vector<long> I_labels;
            vector<size_t> I_dims;
            I_labels.reserve( I_nrVars );
            I_dims.reserve( I_nrVars );
            factorVars[I].reserve( I_nrVars );
            for( size_t _i = 0; _i < I_nrVars; _i++ ) {
                long label;
                is >> label;
                if( is.fail() )
                    DAI_THROWE(INVALID_FACTORGRAPH_FILE,"Cannot read variable labels for " + toString(I) + "'th factor");
                I_labels.push_back( label );
                I_dims.push_back( vars[label].states() );
                factorVars[I].push_back( vars[label] );
            }
            if( verbose >= 3 )
                cout << "  labels: " << I_labels << ", dimensions " << I_dims << endl;

            // add the factor and the labels
            factors.push_back( Factor( VarSet( factorVars[I].begin(), factorVars[I].end(), factorVars[I].size() ), (Real)0 ) );
        }

        // for each factor, read its values
        permutations.reserve( nrFacs );
        for( size_t I = 0; I < nrFacs; I++ ) {
            if( verbose >= 3 )
                cout << "Reading factor " << I << "..." << endl;

            // calculate permutation object, reversing the indexing in factorVars[I] first
            Permute permindex( factorVars[I], true );
            permutations.push_back( permindex );

            // read factor values
            size_t nrNonZeros;
            is >> nrNonZeros;
            if( is.fail() )
                DAI_THROWE(INVALID_FACTORGRAPH_FILE,"Cannot read number of nonzero factor values for " + toString(I) + "'th factor");
            if( verbose >= 3 ) 
                cout << "  number of nonzero values: " << nrNonZeros << endl;
            DAI_ASSERT( nrNonZeros == factors[I].nrStates() );
            for( size_t li = 0; li < nrNonZeros; li++ ) {
                Real val;
                is >> val;
                if( is.fail() )
                    DAI_THROWE(INVALID_FACTORGRAPH_FILE,"Cannot read factor values of " + toString(I) + "'th factor");
                // assign value after calculating its linear index corresponding to the permutation
                if( verbose >= 4 )
                    cout << "  " << li << "'th value " << val << " corresponds with index " << permindex.convertLinearIndex(li) << endl;
                factors[I].set( permindex.convertLinearIndex( li ), val );
            }
        }
        if( verbose >= 3 )
            cout << "variables:" << vars << endl;
        if( verbose >= 3 )
            cout << "factors:" << factors << endl;

        // close file
        is.close();
    } else
        DAI_THROWE(CANNOT_READ_FILE,"Cannot read from file " + std::string(filename));
}


std::vector<std::map<size_t, size_t> > ReadUaiAieEvidenceFile( const char* filename, size_t verbose ) {
    vector<map<size_t, size_t> > evid;

    // open file
    ifstream is;
    string line;
    is.open( filename );
    if( is.is_open() ) {
        // read number of lines
        getline( is, line );
        if( is.fail() || line.size() == 0 )
            DAI_THROWE(INVALID_EVIDENCE_FILE,"Cannot read header line of evidence file");
        if( line[line.size() - 1] == '\r' )
            line.resize( line.size() - 1 ); // for DOS text files
        size_t nrLines = fromString<size_t>( line );
        if( verbose >= 2 )
            cout << "Reading " << nrLines << " evidence file lines..." << endl;

        if( nrLines ) {
            // detect version (pre-2010 or 2010)
            streampos pos = is.tellg();
            getline( is, line );
            if( is.fail() || line.size() == 0 )
                DAI_THROWE(INVALID_EVIDENCE_FILE,"Cannot read second line of evidence file");
            if( line[line.size() - 1] == '\r' )
                line.resize( line.size() - 1 ); // for DOS text files
            vector<string> cols;
            cols = tokenizeString( line, false, " \t" );
            bool oldVersion = true;
            if( cols.size() % 2 )
                oldVersion = false;
            if( verbose >= 2 ) {
                if( oldVersion )
                    cout << "Detected old (2006, 2008) evidence file format" << endl;
                else
                    cout << "Detected new (2010) evidence file format" << endl;
            }
            size_t nrEvid;
            if( oldVersion ) {
                nrEvid = 1;
                is.seekg( 0 );
            } else {
                nrEvid = nrLines;
                is.seekg( pos );
            }
                
            // read all evidence cases
            if( verbose >= 2 )
                cout << "Reading " << nrEvid << " evidence cases..." << endl;
            evid.resize( nrEvid );
            for( size_t i = 0; i < nrEvid; i++ ) {
                // read number of variables
                size_t nrObs;
                is >> nrObs;
                if( is.fail() )
                    DAI_THROWE(INVALID_EVIDENCE_FILE,"Evidence case " + toString(i) + ": Cannot read number of observations");
                if( verbose >= 2 )
                    cout << "Evidence case " << i << ": reading " << nrObs << " observations..." << endl;

                // for each observation, read the variable label and the observed value
                for( size_t j = 0; j < nrObs; j++ ) {
                    size_t label, val;
                    is >> label;
                    if( is.fail() )
                        DAI_THROWE(INVALID_EVIDENCE_FILE,"Evidence case " + toString(i) + ": Cannot read label for " + toString(j) + "'th observed variable");
                    is >> val;
                    if( is.fail() )
                        DAI_THROWE(INVALID_EVIDENCE_FILE,"Evidence case " + toString(i) + ": Cannot read value of " + toString(j) + "'th observed variable");
                    if( verbose >= 3 )
                        cout << "  variable: " << label << ", value: " << val << endl;
                    evid[i][label] = val;
                }
            }
        }

        // close file
        is.close();
    } else
        DAI_THROWE(CANNOT_READ_FILE,"Cannot read from file " + std::string(filename));

    if( evid.size() == 0 )
        evid.resize( 1 );

    return evid;
}


} // end of namespace dai
