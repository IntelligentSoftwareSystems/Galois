/*  This file is part of libDAI - http://www.libdai.org/
 *
 *  Copyright (c) 2006-2011, The libDAI authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
 */


#include <dai/util.h>
#include <boost/random.hpp>

#ifdef WINDOWS
    #include <windows.h>
    #include <boost/math/special_functions/atanh.hpp>  // for atanh
    #include <boost/math/special_functions/log1p.hpp>  // for log1p
    #include <float.h>  // for _isnan
#else
    // Assume POSIX compliant system. We need the following for querying the system time
    #include <sys/time.h>
#endif


#ifdef WINDOWS
double atanh( double x ) {
    return boost::math::atanh( x );
}
double log1p( double x ) {
    return boost::math::log1p( x );
}
#endif


namespace dai {

#if defined CYGWIN
bool isnan( Real x ) {
    return __isnand( x );  // isnan() is a macro in Cygwin (as required by C99)
}
#elif defined WINDOWS
bool isnan( Real x ) {
    return _isnan( x );
}
#else
bool isnan( Real x ) {
    return std::isnan( x );
}
#endif

// Returns user+system time in seconds
double toc() {
#ifdef WINDOWS
    SYSTEMTIME tbuf;
    GetSystemTime(&tbuf);
    return( (double)(tbuf.wSecond + (double)tbuf.wMilliseconds / 1000.0) );
#else
    struct timeval tv;
    struct timezone tz;
    gettimeofday( &tv, &tz );
    return( (double)(tv.tv_sec + (double)tv.tv_usec / 1000000.0) );
#endif
}

/// Type of global random number generator
typedef boost::mt19937 _rnd_gen_type;

/// Global random number generator
_rnd_gen_type _rnd_gen(42U);

/// Uniform distribution with values between 0 and 1 (0 inclusive, 1 exclusive).
boost::uniform_real<Real> _uni_dist(0,1);

/// Normal distribution with mean 0 and standard deviation 1.
boost::normal_distribution<Real> _normal_dist;

/// Global uniform random random number
boost::variate_generator<_rnd_gen_type&, boost::uniform_real<Real> > _uni_rnd(_rnd_gen, _uni_dist);

/// Global random number generator with standard normal distribution
boost::variate_generator<_rnd_gen_type&, boost::normal_distribution<Real> > _normal_rnd(_rnd_gen, _normal_dist);


void rnd_seed( size_t seed ) {
    _rnd_gen.seed( static_cast<unsigned int>(seed) );
    _normal_rnd.distribution().reset(); // needed for clearing the cache used in boost::normal_distribution
}

Real rnd_uniform() {
    return _uni_rnd();
}

Real rnd_stdnormal() {
    return _normal_rnd();
}

int rnd_int( int min, int max ) {
    return (int)floor(_uni_rnd() * (max + 1 - min) + min);
}

std::vector<std::string> tokenizeString( const std::string& s, bool singleDelim, const std::string& delim ) {
    using namespace std;
    vector<string> tokens;

    string::size_type start = 0;
    while( start <= s.size() ) {
        string::size_type end = s.find_first_of( delim, start );
        if( end == string::npos )
            end = s.size();

        if( end == start && !singleDelim ) {
            // skip to next non-delimiter
            start = s.find_first_not_of( delim, start );
            if( start == string::npos )
                break;
        } else { // we found a token
            tokens.push_back( s.substr(start, end - start) );
            start = end + 1;
        }
    }

    return tokens;
}


} // end of namespace dai
