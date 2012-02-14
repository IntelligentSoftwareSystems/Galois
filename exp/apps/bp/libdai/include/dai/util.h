/*  This file is part of libDAI - http://www.libdai.org/
 *
 *  Copyright (c) 2006-2011, The libDAI authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
 */


/// \file
/// \brief Defines general utility functions and adds an abstraction layer for platform-dependent functionality


#ifndef __defined_libdai_util_h
#define __defined_libdai_util_h


#include <string>
#include <vector>
#include <set>
#include <map>
#include <iostream>
#include <boost/foreach.hpp>
#include <boost/functional/hash.hpp>
#include <boost/lexical_cast.hpp>
#include <algorithm>
#include <cerrno>
#include <gmpxx.h>

#include <dai/exceptions.h>


#if defined(WINDOWS)
    #include <boost/tr1/unordered_map.hpp> // only present in boost 1.37 and higher
#elif defined(CYGWIN)
    #include <boost/tr1/unordered_map.hpp> // only present in boost 1.37 and higher
#elif defined(MACOSX)
    #include <boost/tr1/unordered_map.hpp> // only present in boost 1.37 and higher
#else
    #include <tr1/unordered_map> // only present in modern GCC distributions
#endif


/// An alias to the BOOST_FOREACH macro from the boost::foreach library
#define foreach BOOST_FOREACH

#ifdef DAI_DEBUG
/// \brief "Print variable". Prints the text of an expression, followed by its value (only if DAI_DEBUG is defined)
/**
 *  Useful debugging macro to see what your code is doing.
 *  Example: \code DAI_PV(3+4) \endcode
 *  Output: \code 3+4= 7 \endcode
 */
#define DAI_PV(x) do {std::cerr << #x "= " << (x) << std::endl;} while(0)
/// "Debugging message": Prints a message (only if DAI_DEBUG is defined)
#define DAI_DMSG(str) do {std::cerr << str << std::endl;} while(0)
#else
#define DAI_PV(x) do {} while(0)
#define DAI_DMSG(str) do {} while(0)
#endif

/// Macro to write message \a stmt to \c std::cerr if \a props.verbose >= \a n
#define DAI_IFVERB(n, stmt) if(props.verbose>=n) { std::cerr << stmt; }


#ifdef WINDOWS
    /// Returns inverse hyperbolic tangent of argument
    double atanh( double x );

    /// Returns log(1+x)
    double log1p( double x );

    /// Define INFINITY
    #define INFINITY (std::numeric_limits<Real>::infinity())
#endif


namespace dai {


/// Real number (alias for \c double, which could be changed to <tt>long double</tt> if necessary)
typedef double Real;

/// Arbitrary precision integer number
typedef mpz_class BigInt;

/// Safe down-cast of big integer to size_t
inline size_t BigInt_size_t( const BigInt &N ) {
    DAI_ASSERT( N <= std::numeric_limits<std::size_t>::max() );
    return N.get_ui();
}

/// Returns true if argument is NAN (Not A Number)
bool isnan( Real x );

/// Returns logarithm of \a x
inline Real log( Real x ) {
    return std::log(x);
}

/// Returns logarithm of \a x, or 0 if \a x == 0
inline Real log0( Real x ) {
    return x ? std::log(x) : 0;
}

/// Returns exponent of \a x
inline Real exp( Real x ) {
    return std::exp(x);
}

/// Returns \a to the power \a y
inline Real pow( Real x, Real y ) {
    errno = 0;
    Real result = std::pow(x, y);
    DAI_DEBASSERT( errno == 0 );
    return result;
}


/// hash_map is an alias for \c std::tr1::unordered_map.
/** We use the (experimental) TR1 unordered_map implementation included in modern GCC distributions or in boost versions 1.37 and higher.
 */
template <typename T, typename U, typename H = boost::hash<T> >
    class hash_map : public std::tr1::unordered_map<T,U,H> {};


/// Returns wall clock time in seconds
double toc();


/// Returns absolute value of \a t
template<class T>
inline T abs( const T &t ) {
    return (t < 0) ? (-t) : t;
}


/// Sets the random seed
void rnd_seed( size_t seed );

/// Returns a real number, distributed uniformly on [0,1)
Real rnd_uniform();

/// Returns a real number from a standard-normal distribution
Real rnd_stdnormal();

/// Returns a random integer in interval [\a min, \a max]
int rnd_int( int min, int max );

/// Returns a random integer in the half-open interval [0, \a n)
inline int rnd( int n ) {
    return rnd_int( 0, n-1 );
}


/// Converts a variable of type \a T to a \c std::string by using a \c boost::lexical_cast
template<class T>
std::string toString( const T& x ) {
    return boost::lexical_cast<std::string>(x);
}


/// Converts a variable of type std::string to \a T by using a \c boost::lexical_cast
template<class T>
T fromString( const std::string& x ) {
    return boost::lexical_cast<T>(x);
}


/// Writes a \c std::vector<> to a \c std::ostream
template<class T>
std::ostream& operator << (std::ostream& os, const std::vector<T> & x) {
    os << "(";
    for( typename std::vector<T>::const_iterator it = x.begin(); it != x.end(); it++ )
        os << (it != x.begin() ? ", " : "") << *it;
    os << ")";
    return os;
}

/// Writes a \c std::set<> to a \c std::ostream
template<class T>
std::ostream& operator << (std::ostream& os, const std::set<T> & x) {
    os << "{";
    for( typename std::set<T>::const_iterator it = x.begin(); it != x.end(); it++ )
        os << (it != x.begin() ? ", " : "") << *it;
    os << "}";
    return os;
}

/// Writes a \c std::map<> to a \c std::ostream
template<class T1, class T2>
std::ostream& operator << (std::ostream& os, const std::map<T1,T2> & x) {
    os << "{";
    for( typename std::map<T1,T2>::const_iterator it = x.begin(); it != x.end(); it++ )
        os << (it != x.begin() ? ", " : "") << it->first << "->" << it->second;
    os << "}";
    return os;
}

/// Writes a \c std::pair<> to a \c std::ostream
template<class T1, class T2>
std::ostream& operator << (std::ostream& os, const std::pair<T1,T2> & x) {
    os << "(" << x.first << ", " << x.second << ")";
    return os;
}

/// Concatenates two vectors
template<class T>
std::vector<T> concat( const std::vector<T>& u, const std::vector<T>& v ) {
    std::vector<T> w;
    w.reserve( u.size() + v.size() );
    for( size_t i = 0; i < u.size(); i++ )
        w.push_back( u[i] );
    for( size_t i = 0; i < v.size(); i++ )
        w.push_back( v[i] );
    return w;
}

/// Split a string into tokens delimited by one of the characters in \a delim
/** \param s the string to be split into tokens
 *  \param singleDelim if \c true, any single delimiter forms a boundary between two tokens; 
 *         if \c false, a maximal group of consecutive delimiters forms a boundary between two tokens
 *  \param delim delimiter characters
 */
std::vector<std::string> tokenizeString( const std::string& s, bool singleDelim, const std::string& delim="\t\n" );


/// Enumerates different ways of normalizing a probability measure.
/**
 *  - NORMPROB means that the sum of all entries should be 1;
 *  - NORMLINF means that the maximum absolute value of all entries should be 1.
 */
typedef enum { NORMPROB, NORMLINF } ProbNormType;

/// Enumerates different distance measures between probability measures.
/**
 *  - DISTL1 is the \f$\ell_1\f$ distance (sum of absolute values of pointwise difference);
 *  - DISTLINF is the \f$\ell_\infty\f$ distance (maximum absolute value of pointwise difference);
 *  - DISTTV is the total variation distance (half of the \f$\ell_1\f$ distance);
 *  - DISTKL is the Kullback-Leibler distance (\f$\sum_i p_i (\log p_i - \log q_i)\f$).
 *  - DISTHEL is the Hellinger distance (\f$\frac{1}{2}\sum_i (\sqrt{p_i}-\sqrt{q_i})^2\f$).
 */
typedef enum { DISTL1, DISTLINF, DISTTV, DISTKL, DISTHEL } ProbDistType;


} // end of namespace dai


#endif
