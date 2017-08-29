/*  This file is part of libDAI - http://www.libdai.org/
 *
 *  Copyright (c) 2006-2011, The libDAI authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
 */


/// \file
/// \brief Defines TFactor<> and Factor classes which represent factors in probability distributions.


#ifndef __defined_libdai_factor_h
#define __defined_libdai_factor_h


#include <iostream>
#include <functional>
#include <cmath>
#include <dai/prob.h>
#include <dai/varset.h>
#include <dai/index.h>
#include <dai/util.h>


namespace dai {


/// Represents a (probability) factor.
/** Mathematically, a \e factor is a function mapping joint states of some
 *  variables to the nonnegative real numbers.
 *  More formally, denoting a discrete variable with label \f$l\f$ by
 *  \f$x_l\f$ and its state space by \f$X_l = \{0,1,\dots,S_l-1\}\f$,
 *  a factor depending on the variables \f$\{x_l\}_{l\in L}\f$ is
 *  a function \f$f_L : \prod_{l\in L} X_l \to [0,\infty)\f$.
 *
 *  In libDAI, a factor is represented by a TFactor<T> object, which has two
 *  components:
 *  \arg a VarSet, corresponding with the set of variables \f$\{x_l\}_{l\in L}\f$
 *  that the factor depends on;
 *  \arg a TProb, a vector containing the value of the factor for each possible
 *  joint state of the variables.
 *
 *  The factor values are stored in the entries of the TProb in a particular
 *  ordering, which is defined by the one-to-one correspondence of a joint state
 *  in \f$\prod_{l\in L} X_l\f$ with a linear index in
 *  \f$\{0,1,\dots,\prod_{l\in L} S_l-1\}\f$ according to the mapping \f$\sigma\f$
 *  induced by dai::calcLinearState().
 *
 *  \tparam T Should be a scalar that is castable from and to double and should support elementary arithmetic operations.
 *  \todo Define a better fileformat for .fg files (maybe using XML)?
 *  \todo Add support for sparse factors.
 */
template <typename T>
class TFactor {
    private:
        /// Stores the variables on which the factor depends
        VarSet _vs;
        /// Stores the factor values
        TProb<T> _p;

    public:
    /// \name Constructors and destructors
    //@{
        /// Constructs factor depending on no variables with value \a p
        TFactor ( T p = 1 ) : _vs(), _p(1,p) {}

        /// Constructs factor depending on the variable \a v with uniform distribution
        TFactor( const Var &v ) : _vs(v), _p(v.states()) {}

        /// Constructs factor depending on variables in \a vars with uniform distribution
        TFactor( const VarSet& vars ) : _vs(vars), _p() {
            _p = TProb<T>( BigInt_size_t( _vs.nrStates() ) );
        }

        /// Constructs factor depending on variables in \a vars with all values set to \a p
        TFactor( const VarSet& vars, T p ) : _vs(vars), _p() {
            _p = TProb<T>( BigInt_size_t( _vs.nrStates() ), p );
        }

        /// Constructs factor depending on variables in \a vars, copying the values from a std::vector<>
        /** \tparam S Type of values of \a x
         *  \param vars contains the variables that the new factor should depend on.
         *  \param x Vector with values to be copied.
         */
        template<typename S>
        TFactor( const VarSet& vars, const std::vector<S> &x ) : _vs(vars), _p() {
            DAI_ASSERT( x.size() == vars.nrStates() );
            _p = TProb<T>( x.begin(), x.end(), x.size() );
        }

        /// Constructs factor depending on variables in \a vars, copying the values from an array
        /** \param vars contains the variables that the new factor should depend on.
         *  \param p Points to array of values to be added.
         */
        TFactor( const VarSet& vars, const T* p ) : _vs(vars), _p() {
            size_t N = BigInt_size_t( _vs.nrStates() );
            _p = TProb<T>( p, p + N, N );
        }

        /// Constructs factor depending on variables in \a vars, copying the values from \a p
        TFactor( const VarSet& vars, const TProb<T> &p ) : _vs(vars), _p(p) {
            DAI_ASSERT( _vs.nrStates() == _p.size() );
        }

        /// Constructs factor depending on variables in \a vars, permuting the values given in \a p accordingly
        TFactor( const std::vector<Var> &vars, const std::vector<T> &p ) : _vs(vars.begin(), vars.end(), vars.size()), _p(p.size()) {
            BigInt nrStates = 1;
            for( size_t i = 0; i < vars.size(); i++ )
                nrStates *= vars[i].states();
            DAI_ASSERT( nrStates == p.size() );
            Permute permindex(vars);
            for( size_t li = 0; li < p.size(); ++li )
                _p.set( permindex.convertLinearIndex(li), p[li] );
        }
    //@}

    /// \name Get/set individual entries
    //@{
        /// Sets \a i 'th entry to \a val
        void set( size_t i, T val ) { _p.set( i, val ); }

        /// Gets \a i 'th entry
        T get( size_t i ) const { return _p[i]; }
    //@}

    /// \name Queries
    //@{
        /// Returns constant reference to value vector
        const TProb<T>& p() const { return _p; }

        /// Returns reference to value vector
        TProb<T>& p() { return _p; }

        /// Returns a copy of the \a i 'th entry of the value vector
        T operator[] (size_t i) const { return _p[i]; }

        /// Returns constant reference to variable set (i.e., the variables on which the factor depends)
        const VarSet& vars() const { return _vs; }

        /// Returns reference to variable set (i.e., the variables on which the factor depends)
        VarSet& vars() { return _vs; }

        /// Returns the number of possible joint states of the variables on which the factor depends, \f$\prod_{l\in L} S_l\f$
        /** \note This is equal to the length of the value vector.
         */
        size_t nrStates() const { return _p.size(); }

        /// Returns the Shannon entropy of \c *this, \f$-\sum_i p_i \log p_i\f$
        T entropy() const { return _p.entropy(); }

        /// Returns maximum of all values
        T max() const { return _p.max(); }

        /// Returns minimum of all values
        T min() const { return _p.min(); }

        /// Returns sum of all values
        T sum() const { return _p.sum(); }
        
        /// Returns sum of absolute values
        T sumAbs() const { return _p.sumAbs(); }

        /// Returns maximum absolute value of all values
        T maxAbs() const { return _p.maxAbs(); }

        /// Returns \c true if one or more values are NaN
        bool hasNaNs() const { return _p.hasNaNs(); }

        /// Returns \c true if one or more values are negative
        bool hasNegatives() const { return _p.hasNegatives(); }

        /// Returns strength of this factor (between variables \a i and \a j), as defined in eq. (52) of [\ref MoK07b]
        T strength( const Var &i, const Var &j ) const;

        /// Comparison
        bool operator==( const TFactor<T>& y ) const {
            return (_vs == y._vs) && (_p == y._p);
        }
    //@}

    /// \name Unary transformations
    //@{
        /// Returns negative of \c *this
        TFactor<T> operator- () const { 
            // Note: the alternative (shorter) way of implementing this,
            //   return TFactor<T>( _vs, _p.abs() );
            // is slower because it invokes the copy constructor of TProb<T>
            TFactor<T> x;
            x._vs = _vs;
            x._p = -_p;
            return x;
        }

        /// Returns pointwise absolute value
        TFactor<T> abs() const {
            TFactor<T> x;
            x._vs = _vs;
            x._p = _p.abs();
            return x;
        }

        /// Returns pointwise exponent
        TFactor<T> exp() const {
            TFactor<T> x;
            x._vs = _vs;
            x._p = _p.exp();
            return x;
        }

        /// Returns pointwise logarithm
        /** If \a zero == \c true, uses <tt>log(0)==0</tt>; otherwise, <tt>log(0)==-Inf</tt>.
         */
        TFactor<T> log(bool zero=false) const {
            TFactor<T> x;
            x._vs = _vs;
            x._p = _p.log(zero);
            return x;
        }

        /// Returns pointwise inverse
        /** If \a zero == \c true, uses <tt>1/0==0</tt>; otherwise, <tt>1/0==Inf</tt>.
         */
        TFactor<T> inverse(bool zero=true) const {
            TFactor<T> x;
            x._vs = _vs;
            x._p = _p.inverse(zero);
            return x;
        }

        /// Returns normalized copy of \c *this, using the specified norm
        /** \throw NOT_NORMALIZABLE if the norm is zero
         */
        TFactor<T> normalized( ProbNormType norm=NORMPROB ) const {
            TFactor<T> x;
            x._vs = _vs;
            x._p = _p.normalized( norm );
            return x;
        }
    //@}

    /// \name Unary operations
    //@{
        /// Draws all values i.i.d. from a uniform distribution on [0,1)
        TFactor<T>& randomize() { _p.randomize(); return *this; }

        /// Sets all values to \f$1/n\f$ where \a n is the number of states
        TFactor<T>& setUniform() { _p.setUniform(); return *this; }

        /// Applies absolute value pointwise
        TFactor<T>& takeAbs() { _p.takeAbs(); return *this; }

        /// Applies exponent pointwise
        TFactor<T>& takeExp() { _p.takeExp(); return *this; }

        /// Applies logarithm pointwise
        /** If \a zero == \c true, uses <tt>log(0)==0</tt>; otherwise, <tt>log(0)==-Inf</tt>.
         */
        TFactor<T>& takeLog( bool zero = false ) { _p.takeLog(zero); return *this; }

        /// Normalizes factor using the specified norm
        /** \throw NOT_NORMALIZABLE if the norm is zero
         */
        T normalize( ProbNormType norm=NORMPROB ) { return _p.normalize( norm ); }
    //@}

    /// \name Operations with scalars
    //@{
        /// Sets all values to \a x
        TFactor<T>& fill (T x) { _p.fill( x ); return *this; }

        /// Adds scalar \a x to each value
        TFactor<T>& operator+= (T x) { _p += x; return *this; }

        /// Subtracts scalar \a x from each value
        TFactor<T>& operator-= (T x) { _p -= x; return *this; }

        /// Multiplies each value with scalar \a x
        TFactor<T>& operator*= (T x) { _p *= x; return *this; }

        /// Divides each entry by scalar \a x
        TFactor<T>& operator/= (T x) { _p /= x; return *this; }

        /// Raises values to the power \a x
        TFactor<T>& operator^= (T x) { _p ^= x; return *this; }
    //@}

    /// \name Transformations with scalars
    //@{
        /// Returns sum of \c *this and scalar \a x
        TFactor<T> operator+ (T x) const {
            // Note: the alternative (shorter) way of implementing this,
            //   TFactor<T> result(*this);
            //   result._p += x;
            // is slower because it invokes the copy constructor of TFactor<T>
            TFactor<T> result;
            result._vs = _vs;
            result._p = p() + x;
            return result;
        }

        /// Returns difference of \c *this and scalar \a x
        TFactor<T> operator- (T x) const {
            TFactor<T> result;
            result._vs = _vs;
            result._p = p() - x;
            return result;
        }

        /// Returns product of \c *this with scalar \a x
        TFactor<T> operator* (T x) const {
            TFactor<T> result;
            result._vs = _vs;
            result._p = p() * x;
            return result;
        }

        /// Returns quotient of \c *this with scalar \a x
        TFactor<T> operator/ (T x) const {
            TFactor<T> result;
            result._vs = _vs;
            result._p = p() / x;
            return result;
        }

        /// Returns \c *this raised to the power \a x
        TFactor<T> operator^ (T x) const {
            TFactor<T> result;
            result._vs = _vs;
            result._p = p() ^ x;
            return result;
        }
    //@}

    /// \name Operations with other factors
    //@{
        /// Applies binary operation \a op on two factors, \c *this and \a g
        /** \tparam binOp Type of function object that accepts two arguments of type \a T and outputs a type \a T
         *  \param g Right operand
         *  \param op Operation of type \a binOp
         */
        template<typename binOp> TFactor<T>& binaryOp( const TFactor<T> &g, binOp op ) {
            if( _vs == g._vs ) // optimize special case
                _p.pwBinaryOp( g._p, op );
            else {
                TFactor<T> f(*this); // make a copy
                _vs |= g._vs;
                size_t N = BigInt_size_t( _vs.nrStates() );

                IndexFor i_f( f._vs, _vs );
                IndexFor i_g( g._vs, _vs );

                _p.p().clear();
                _p.p().reserve( N );
                for( size_t i = 0; i < N; i++, ++i_f, ++i_g )
                    _p.p().push_back( op( f._p[i_f], g._p[i_g] ) );
            }
            return *this;
        }

        /// Adds \a g to \c *this
        /** The sum of two factors is defined as follows: if
         *  \f$f : \prod_{l\in L} X_l \to [0,\infty)\f$ and \f$g : \prod_{m\in M} X_m \to [0,\infty)\f$, then
         *  \f[f+g : \prod_{l\in L\cup M} X_l \to [0,\infty) : x \mapsto f(x_L) + g(x_M).\f]
         */
        TFactor<T>& operator+= (const TFactor<T>& g) { return binaryOp( g, std::plus<T>() ); }

        /// Subtracts \a g from \c *this
        /** The difference of two factors is defined as follows: if
         *  \f$f : \prod_{l\in L} X_l \to [0,\infty)\f$ and \f$g : \prod_{m\in M} X_m \to [0,\infty)\f$, then
         *  \f[f-g : \prod_{l\in L\cup M} X_l \to [0,\infty) : x \mapsto f(x_L) - g(x_M).\f]
         */
        TFactor<T>& operator-= (const TFactor<T>& g) { return binaryOp( g, std::minus<T>() ); }

        /// Multiplies \c *this with \a g
        /** The product of two factors is defined as follows: if
         *  \f$f : \prod_{l\in L} X_l \to [0,\infty)\f$ and \f$g : \prod_{m\in M} X_m \to [0,\infty)\f$, then
         *  \f[fg : \prod_{l\in L\cup M} X_l \to [0,\infty) : x \mapsto f(x_L) g(x_M).\f]
         */
        TFactor<T>& operator*= (const TFactor<T>& g) { return binaryOp( g, std::multiplies<T>() ); }

        /// Divides \c *this by \a g (where division by zero yields zero)
        /** The quotient of two factors is defined as follows: if
         *  \f$f : \prod_{l\in L} X_l \to [0,\infty)\f$ and \f$g : \prod_{m\in M} X_m \to [0,\infty)\f$, then
         *  \f[\frac{f}{g} : \prod_{l\in L\cup M} X_l \to [0,\infty) : x \mapsto \frac{f(x_L)}{g(x_M)}.\f]
         */
        TFactor<T>& operator/= (const TFactor<T>& g) { return binaryOp( g, fo_divides0<T>() ); }
    //@}

    /// \name Transformations with other factors
    //@{
        /// Returns result of applying binary operation \a op on two factors, \c *this and \a g
        /** \tparam binOp Type of function object that accepts two arguments of type \a T and outputs a type \a T
         *  \param g Right operand
         *  \param op Operation of type \a binOp
         */
        template<typename binOp> TFactor<T> binaryTr( const TFactor<T> &g, binOp op ) const {
            // Note that to prevent a copy to be made, it is crucial 
            // that the result is declared outside the if-else construct.
            TFactor<T> result;
            if( _vs == g._vs ) { // optimize special case
                result._vs = _vs;
                result._p = _p.pwBinaryTr( g._p, op );
            } else {
                result._vs = _vs | g._vs;
                size_t N = BigInt_size_t( result._vs.nrStates() );

                IndexFor i_f( _vs, result.vars() );
                IndexFor i_g( g._vs, result.vars() );

                result._p.p().clear();
                result._p.p().reserve( N );
                for( size_t i = 0; i < N; i++, ++i_f, ++i_g )
                    result._p.p().push_back( op( _p[i_f], g[i_g] ) );
            }
            return result;
        }

        /// Returns sum of \c *this and \a g
        /** The sum of two factors is defined as follows: if
         *  \f$f : \prod_{l\in L} X_l \to [0,\infty)\f$ and \f$g : \prod_{m\in M} X_m \to [0,\infty)\f$, then
         *  \f[f+g : \prod_{l\in L\cup M} X_l \to [0,\infty) : x \mapsto f(x_L) + g(x_M).\f]
         */
        TFactor<T> operator+ (const TFactor<T>& g) const {
            return binaryTr(g,std::plus<T>());
        }

        /// Returns \c *this minus \a g
        /** The difference of two factors is defined as follows: if
         *  \f$f : \prod_{l\in L} X_l \to [0,\infty)\f$ and \f$g : \prod_{m\in M} X_m \to [0,\infty)\f$, then
         *  \f[f-g : \prod_{l\in L\cup M} X_l \to [0,\infty) : x \mapsto f(x_L) - g(x_M).\f]
         */
        TFactor<T> operator- (const TFactor<T>& g) const {
            return binaryTr(g,std::minus<T>());
        }

        /// Returns product of \c *this with \a g
        /** The product of two factors is defined as follows: if
         *  \f$f : \prod_{l\in L} X_l \to [0,\infty)\f$ and \f$g : \prod_{m\in M} X_m \to [0,\infty)\f$, then
         *  \f[fg : \prod_{l\in L\cup M} X_l \to [0,\infty) : x \mapsto f(x_L) g(x_M).\f]
         */
        TFactor<T> operator* (const TFactor<T>& g) const {
            return binaryTr(g,std::multiplies<T>());
        }

        /// Returns quotient of \c *this by \a f (where division by zero yields zero)
        /** The quotient of two factors is defined as follows: if
         *  \f$f : \prod_{l\in L} X_l \to [0,\infty)\f$ and \f$g : \prod_{m\in M} X_m \to [0,\infty)\f$, then
         *  \f[\frac{f}{g} : \prod_{l\in L\cup M} X_l \to [0,\infty) : x \mapsto \frac{f(x_L)}{g(x_M)}.\f]
         */
        TFactor<T> operator/ (const TFactor<T>& g) const {
            return binaryTr(g,fo_divides0<T>());
        }
    //@}

    /// \name Miscellaneous operations
    //@{
        /// Returns a slice of \c *this, where the subset \a vars is in state \a varsState
        /** \pre \a vars sould be a subset of vars()
         *  \pre \a varsState < vars.nrStates()
         *
         *  The result is a factor that depends on the variables of *this except those in \a vars,
         *  obtained by setting the variables in \a vars to the joint state specified by the linear index
         *  \a varsState. Formally, if \c *this corresponds with the factor \f$f : \prod_{l\in L} X_l \to [0,\infty)\f$,
         *  \f$M \subset L\f$ corresponds with \a vars and \a varsState corresponds with a mapping \f$s\f$ that
         *  maps a variable \f$x_m\f$ with \f$m\in M\f$ to its state \f$s(x_m) \in X_m\f$, then the slice
         *  returned corresponds with the factor \f$g : \prod_{l \in L \setminus M} X_l \to [0,\infty)\f$
         *  defined by \f$g(\{x_l\}_{l\in L \setminus M}) = f(\{x_l\}_{l\in L \setminus M}, \{s(x_m)\}_{m\in M})\f$.
         */
        TFactor<T> slice( const VarSet& vars, size_t varsState ) const; 

        /// Embeds this factor in a larger VarSet
        /** \pre vars() should be a subset of \a vars 
         *
         *  If *this corresponds with \f$f : \prod_{l\in L} X_l \to [0,\infty)\f$ and \f$L \subset M\f$, then
         *  the embedded factor corresponds with \f$g : \prod_{m\in M} X_m \to [0,\infty) : x \mapsto f(x_L)\f$.
         */
        TFactor<T> embed(const VarSet & vars) const {
            DAI_ASSERT( vars >> _vs );
            if( _vs == vars )
                return *this;
            else
                return (*this) * TFactor<T>(vars / _vs, (T)1);
        }

        /// Returns marginal on \a vars, obtained by summing out all variables except those in \a vars, and normalizing the result if \a normed == \c true
        TFactor<T> marginal(const VarSet &vars, bool normed=true) const;

        /// Returns max-marginal on \a vars, obtained by maximizing all variables except those in \a vars, and normalizing the result if \a normed == \c true
        TFactor<T> maxMarginal(const VarSet &vars, bool normed=true) const;
    //@}
};


template<typename T> TFactor<T> TFactor<T>::slice( const VarSet& vars, size_t varsState ) const {
    DAI_ASSERT( vars << _vs );
    VarSet varsrem = _vs / vars;
    TFactor<T> result( varsrem, T(0) );

    // OPTIMIZE ME
    IndexFor i_vars (vars, _vs);
    IndexFor i_varsrem (varsrem, _vs);
    for( size_t i = 0; i < nrStates(); i++, ++i_vars, ++i_varsrem )
        if( (size_t)i_vars == varsState )
            result.set( i_varsrem, _p[i] );

    return result;
}


template<typename T> TFactor<T> TFactor<T>::marginal(const VarSet &vars, bool normed) const {
    VarSet res_vars = vars & _vs;

    TFactor<T> res( res_vars, 0.0 );

    IndexFor i_res( res_vars, _vs );
    for( size_t i = 0; i < _p.size(); i++, ++i_res )
        res.set( i_res, res[i_res] + _p[i] );

    if( normed )
        res.normalize( NORMPROB );

    return res;
}


template<typename T> TFactor<T> TFactor<T>::maxMarginal(const VarSet &vars, bool normed) const {
    VarSet res_vars = vars & _vs;

    TFactor<T> res( res_vars, 0.0 );

    IndexFor i_res( res_vars, _vs );
    for( size_t i = 0; i < _p.size(); i++, ++i_res )
        if( _p[i] > res._p[i_res] )
            res.set( i_res, _p[i] );

    if( normed )
        res.normalize( NORMPROB );

    return res;
}


template<typename T> T TFactor<T>::strength( const Var &i, const Var &j ) const {
    DAI_DEBASSERT( _vs.contains( i ) );
    DAI_DEBASSERT( _vs.contains( j ) );
    DAI_DEBASSERT( i != j );
    VarSet ij(i, j);

    T max = 0.0;
    for( size_t alpha1 = 0; alpha1 < i.states(); alpha1++ )
        for( size_t alpha2 = 0; alpha2 < i.states(); alpha2++ )
            if( alpha2 != alpha1 )
                for( size_t beta1 = 0; beta1 < j.states(); beta1++ )
                    for( size_t beta2 = 0; beta2 < j.states(); beta2++ )
                        if( beta2 != beta1 ) {
                            size_t as = 1, bs = 1;
                            if( i < j )
                                bs = i.states();
                            else
                                as = j.states();
                            T f1 = slice( ij, alpha1 * as + beta1 * bs ).p().divide( slice( ij, alpha2 * as + beta1 * bs ).p() ).max();
                            T f2 = slice( ij, alpha2 * as + beta2 * bs ).p().divide( slice( ij, alpha1 * as + beta2 * bs ).p() ).max();
                            T f = f1 * f2;
                            if( f > max )
                                max = f;
                        }

    return std::tanh( 0.25 * std::log( max ) );
}


/// Writes a factor to an output stream
/** \relates TFactor
 */
template<typename T> std::ostream& operator<< (std::ostream& os, const TFactor<T>& f) {
    os << "(" << f.vars() << ", (";
    for( size_t i = 0; i < f.nrStates(); i++ )
        os << (i == 0 ? "" : ", ") << f[i];
    os << "))";
    return os;
}


/// Returns distance between two factors \a f and \a g, according to the distance measure \a dt
/** \relates TFactor
 *  \pre f.vars() == g.vars()
 */
template<typename T> T dist( const TFactor<T> &f, const TFactor<T> &g, ProbDistType dt ) {
    if( f.vars().empty() || g.vars().empty() )
        return -1;
    else {
        DAI_DEBASSERT( f.vars() == g.vars() );
        return dist( f.p(), g.p(), dt );
    }
}


/// Returns the pointwise maximum of two factors
/** \relates TFactor
 *  \pre f.vars() == g.vars()
 */
template<typename T> TFactor<T> max( const TFactor<T> &f, const TFactor<T> &g ) {
    DAI_ASSERT( f.vars() == g.vars() );
    return TFactor<T>( f.vars(), max( f.p(), g.p() ) );
}


/// Returns the pointwise minimum of two factors
/** \relates TFactor
 *  \pre f.vars() == g.vars()
 */
template<typename T> TFactor<T> min( const TFactor<T> &f, const TFactor<T> &g ) {
    DAI_ASSERT( f.vars() == g.vars() );
    return TFactor<T>( f.vars(), min( f.p(), g.p() ) );
}


/// Calculates the mutual information between the two variables that \a f depends on, under the distribution given by \a f
/** \relates TFactor
 *  \pre f.vars().size() == 2
 */
template<typename T> T MutualInfo(const TFactor<T> &f) {
    DAI_ASSERT( f.vars().size() == 2 );
    VarSet::const_iterator it = f.vars().begin();
    Var i = *it; it++; Var j = *it;
    TFactor<T> projection = f.marginal(i) * f.marginal(j);
    return dist( f.normalized(), projection, DISTKL );
}


/// Represents a factor with values of type dai::Real.
typedef TFactor<Real> Factor;


/// Returns a binary unnormalized single-variable factor \f$ \exp(hx) \f$ where \f$ x = \pm 1 \f$
/** \param x Variable (should be binary)
 *  \param h Field strength
 */
Factor createFactorIsing( const Var &x, Real h );


/// Returns a binary unnormalized pairwise factor \f$ \exp(J x_1 x_2) \f$ where \f$ x_1, x_2 = \pm 1 \f$
/** \param x1 First variable (should be binary)
 *  \param x2 Second variable (should be binary)
 *  \param J Coupling strength
 */
Factor createFactorIsing( const Var &x1, const Var &x2, Real J );


/// Returns a random factor on the variables \a vs with strength \a beta
/** Each entry are set by drawing a normally distributed random with mean
 *  0 and standard-deviation \a beta, and taking its exponent.
 *  \param vs Variables
 *  \param beta Factor strength (inverse temperature)
 */
Factor createFactorExpGauss( const VarSet &vs, Real beta );


/// Returns a pairwise Potts factor \f$ \exp( J \delta_{x_1, x_2} ) \f$
/** \param x1 First variable
 *  \param x2 Second variable (should have the same number of states as \a x1)
 *  \param J  Factor strength
 */
Factor createFactorPotts( const Var &x1, const Var &x2, Real J );


/// Returns a Kronecker delta point mass
/** \param v Variable
 *  \param state The state of \a v that should get value 1
 */
Factor createFactorDelta( const Var &v, size_t state );


/// Returns a Kronecker delta point mass
/** \param vs Set of variables
 *  \param state The state of \a vs that should get value 1
 */
Factor createFactorDelta( const VarSet& vs, size_t state );


} // end of namespace dai


#endif
