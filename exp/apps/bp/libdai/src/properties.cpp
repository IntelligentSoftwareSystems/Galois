/*  This file is part of libDAI - http://www.libdai.org/
 *
 *  Copyright (c) 2006-2011, The libDAI authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
 */


#include <iostream>
#include <dai/properties.h>
#include <dai/exceptions.h>


namespace dai {


std::ostream& operator<< (std::ostream & os, const Property & p) {
    os << p.first << "=";
    if( p.second.type() == typeid(size_t) )
        os << boost::any_cast<size_t>(p.second);
    else if( p.second.type() == typeid(int) )
        os << boost::any_cast<int>(p.second);
    else if( p.second.type() == typeid(std::string) )
        os << boost::any_cast<std::string>(p.second);
    else if( p.second.type() == typeid(double) )
        os << boost::any_cast<double>(p.second);
    else if( p.second.type() == typeid(long double) )
        os << boost::any_cast<long double>(p.second);
    else if( p.second.type() == typeid(bool) )
        os << boost::any_cast<bool>(p.second);
    else if( p.second.type() == typeid(PropertySet) )
        os << boost::any_cast<PropertySet>(p.second);
    else
        DAI_THROW(UNKNOWN_PROPERTY_TYPE);
    return( os );
}


/// Writes a PropertySet object to an output stream
std::ostream& operator<< (std::ostream & os, const PropertySet & ps) {
    os << "[";
    for( PropertySet::const_iterator p = ps.begin(); p != ps.end(); p++ ) {
        if( p != ps.begin() )
            os << ",";
        os << (Property)*p;
    }
    os << "]";
    return os;
}


/// Reads a PropertySet object from an input stream, storing values as strings
std::istream& operator>> (std::istream& is, PropertySet & ps) {
    ps = PropertySet();

    std::string s;
    is >> s;

    // Check whether s is of the form "[.*]"
    if( (s.length() < 2) || (s.at(0) != '[') || (s.at(s.length()-1)) != ']' )
        DAI_THROWE(MALFORMED_PROPERTY,"Malformed PropertySet: " + s);

    size_t N = s.length() - 1;
    for( size_t token_start = 1; token_start < N; ) {
        size_t token_end;

        // scan until '=' is found
        for( token_end = token_start + 1; token_end < N; token_end++ )
            if( s[token_end] == '=' )
                break;
        // we found a key
        std::string key = s.substr(token_start, token_end - token_start);
        if( token_end == N )
            DAI_THROWE(MALFORMED_PROPERTY,"Malformed Property: " + key);

        token_start = token_end + 1;
        // scan until matching ',' is found
        int level = 0;
        for( token_end = token_start; token_end < N; token_end++ ) {
            if( s[token_end] == '[' )
                level++;
            else if( s[token_end] == ']' )
                level--;
            else if( (s[token_end] == ',') && (level == 0) )
                break;
        }
        if( !(level == 0) )
            DAI_THROWE(MALFORMED_PROPERTY,"Malformed Property: " + s.substr(token_start, token_end - token_start));
        // we found a vlue
        std::string value = s.substr(token_start, token_end - token_start);

        // store the key,value pair
        ps.set(key,value);

        // go on with the next one
        token_start = token_end + 1;
    }

    return is;
}


} // end of namespace dai
