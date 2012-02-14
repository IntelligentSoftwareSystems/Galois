/*  This file is part of libDAI - http://www.libdai.org/
 *
 *  Copyright (c) 2006-2011, The libDAI authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
 */


/// \file
/// \brief Defines ExactInf class, which can be used for exact inference on small factor graphs.


#ifndef __defined_libdai_exactinf_h
#define __defined_libdai_exactinf_h


#include <dai/daialg.h>
#include <dai/properties.h>
#include <dai/factorgraph.h>
#include <dai/enum.h>


namespace dai {


/// Exact inference algorithm using brute force enumeration (mainly useful for testing purposes)
/** Inference is done simply by multiplying all factors together into one large factor,
 *  and then calculating marginals and partition sum from the product.
 *  \note This inference method can easily exhaust all available memory; in that case, one
 *  may try the JTree class instead.
 */
class ExactInf : public DAIAlgFG {
    public:
        /// Parameters for ExactInf
        struct Properties {
            /// Verbosity (amount of output sent to stderr)
            size_t verbose;
        } props;

    private:
        /// All single variable marginals
        std::vector<Factor> _beliefsV;
        /// All factor variable marginals
        std::vector<Factor> _beliefsF;
        /// Logarithm of partition sum
        Real                _logZ;

    public:
    /// \name Constructors/destructors
    //@{
        /// Default constructor
        ExactInf() : DAIAlgFG(), props(), _beliefsV(), _beliefsF(), _logZ(0) {}

        /// Construct from FactorGraph \a fg and PropertySet \a opts
        /** \param fg Factor graph.
         *  \param opts Parameters @see Properties
         */
        ExactInf( const FactorGraph &fg, const PropertySet &opts ) : DAIAlgFG(fg), props(), _beliefsV(), _beliefsF(), _logZ() {
            setProperties( opts );
            construct();
        }
    //@}

    /// \name General InfAlg interface
    //@{
        virtual ExactInf* clone() const { return new ExactInf(*this); }
        virtual ExactInf* construct( const FactorGraph &fg, const PropertySet &opts ) const { return new ExactInf( fg, opts ); }
        virtual std::string name() const { return "EXACT"; }
        virtual Factor belief( const Var &v ) const { return beliefV( findVar( v ) ); }
        virtual Factor belief( const VarSet &vs ) const;
        virtual Factor beliefV( size_t i ) const { return _beliefsV[i]; }
        virtual Factor beliefF( size_t I ) const { return _beliefsF[I]; }
        virtual std::vector<Factor> beliefs() const;
        virtual Real logZ() const { return _logZ; }
        /** \note The complexity of this calculation is exponential in the number of variables.
         */
        std::vector<std::size_t> findMaximum() const;
        virtual void init();
        virtual void init( const VarSet &/*ns*/ ) {}
        virtual Real run();
        virtual Real maxDiff() const { DAI_THROW(NOT_IMPLEMENTED); return 0.0; }
        virtual size_t Iterations() const { DAI_THROW(NOT_IMPLEMENTED); return 0; }
        virtual void setProperties( const PropertySet &opts );
        virtual PropertySet getProperties() const;
        virtual std::string printProperties() const;
    //@}

    /// \name Additional interface specific for ExactInf
    //@{
        /// Calculates marginal probability distribution for variables \a vs
        /** \note The complexity of this calculation is exponential in the number of variables.
         */
        Factor calcMarginal( const VarSet &vs ) const;
    //@}

    private:
        /// Helper function for constructors
        void construct();
};


} // end of namespace dai


#endif
