/*  This file is part of libDAI - http://www.libdai.org/
 *
 *  Copyright (c) 2006-2011, The libDAI authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
 */


#include <string>
#include <fstream>
#include <dai/alldai.h>
#include <dai/properties.h>
#include <dai/exceptions.h>


namespace dai {


using namespace std;


class _builtinInfAlgs : public std::map<std::string, InfAlg *> { 
    public: 
        _builtinInfAlgs() {
            operator[]( ExactInf().name() ) = new ExactInf;
#ifdef DAI_WITH_BP
            operator[]( BP().name() ) = new BP;
#endif
#ifdef DAI_WITH_FBP
            operator[]( FBP().name() ) = new FBP;
#endif
#ifdef DAI_WITH_TRWBP
            operator[]( TRWBP().name() ) = new TRWBP;
#endif
#ifdef DAI_WITH_MF
            operator[]( MF().name() ) = new MF;
#endif
#ifdef DAI_WITH_HAK
            operator[]( HAK().name() ) = new HAK;
#endif
#ifdef DAI_WITH_LC
            operator[]( LC().name() ) = new LC;
#endif
#ifdef DAI_WITH_TREEEP
            operator[]( TreeEP().name() ) = new TreeEP;
#endif
#ifdef DAI_WITH_JTREE
            operator[]( JTree().name() ) = new JTree;
#endif
#ifdef DAI_WITH_MR
            operator[]( MR().name() ) = new MR;
#endif
#ifdef DAI_WITH_GIBBS
            operator[]( Gibbs().name() ) = new Gibbs;
#endif
#ifdef DAI_WITH_CBP
            operator[]( CBP().name() ) = new CBP;
#endif
#ifdef DAI_WITH_DECMAP
            operator[]( DecMAP().name() ) = new DecMAP;
#endif
        }

        ~_builtinInfAlgs() {
            for( iterator it = begin(); it != end(); it++ )
                delete it->second;
        }
};


static _builtinInfAlgs allBuiltinInfAlgs;


std::map<std::string, InfAlg *>& builtinInfAlgs() {
    return allBuiltinInfAlgs;
}


std::set<std::string> builtinInfAlgNames() {
    std::set<std::string> algNames;
    for( _builtinInfAlgs::const_iterator it = allBuiltinInfAlgs.begin(); it != allBuiltinInfAlgs.end(); it++ )
        algNames.insert( it->first );
    return algNames;
}


InfAlg *newInfAlg( const std::string &name, const FactorGraph &fg, const PropertySet &opts ) {
    _builtinInfAlgs::const_iterator i = allBuiltinInfAlgs.find( name );
    if( i == allBuiltinInfAlgs.end() )
        DAI_THROWE(UNKNOWN_DAI_ALGORITHM, "Unknown inference algorithm: " + name);
    InfAlg *ia = i->second;
    return ia->construct( fg, opts );
}


InfAlg *newInfAlgFromString( const std::string &nameOpts, const FactorGraph &fg ) {
    pair<string,PropertySet> no = parseNameProperties( nameOpts );
    return newInfAlg( no.first, fg, no.second );
}


InfAlg *newInfAlgFromString( const std::string &nameOpts, const FactorGraph &fg, const std::map<std::string,std::string> &aliases ) {
    pair<string,PropertySet> no = parseNameProperties( nameOpts, aliases );
    return newInfAlg( no.first, fg, no.second );
}


std::pair<std::string, PropertySet> parseNameProperties( const std::string &s ) {
    string::size_type pos = s.find_first_of('[');
    string name;
    PropertySet opts;
    if( pos == string::npos ) {
        name = s;
    } else {
        name = s.substr(0,pos);

        stringstream ss;
        ss << s.substr(pos,s.length());
        ss >> opts;
    }
    return make_pair(name,opts);
}


std::pair<std::string, PropertySet> parseNameProperties( const std::string &s, const std::map<std::string,std::string> &aliases ) {
    // break string into method[properties]
    pair<string,PropertySet> ps = parseNameProperties(s);
    bool looped = false;

    // as long as 'method' is an alias, update:
    while( aliases.find(ps.first) != aliases.end() && !looped ) {
        string astr = aliases.find(ps.first)->second;
        pair<string,PropertySet> aps = parseNameProperties(astr);
        if( aps.first == ps.first )
            looped = true;
        // override aps properties by ps properties
        aps.second.set( ps.second );
        // replace ps by aps
        ps = aps;
        // repeat until method name == alias name ('looped'), or
        // there is no longer an alias 'method'
    }

    return ps;
}


std::map<std::string,std::string> readAliasesFile( const std::string &filename ) {
    // Read aliases
    map<string,string> result;
    ifstream infile;
    infile.open( filename.c_str() );
    if( infile.is_open() ) {
        while( true ) {
            string line;
            getline( infile,line );
            if( infile.fail() )
                break;
            if( (!line.empty()) && (line[0] != '#') ) {
                string::size_type pos = line.find(':',0);
                if( pos == string::npos )
                    DAI_THROWE(INVALID_ALIAS,"Invalid alias '" + line + "'");
                else {
                    string::size_type posl = line.substr(0, pos).find_last_not_of(" \t");
                    string key = line.substr(0, posl + 1);
                    string::size_type posr = line.substr(pos + 1, line.length()).find_first_not_of(" \t");
                    string val = line.substr(pos + 1 + posr, line.length());
                    result[key] = val;
                }
            }
        }
        infile.close();
    } else
        DAI_THROWE(CANNOT_READ_FILE,"Error opening aliases file " + filename);
    return result;
}


} // end of namespace dai
