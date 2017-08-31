/*  This file is part of libDAI - http://www.libdai.org/
 *
 *  Copyright (c) 2006-2011, The libDAI authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
 */


/// \file
/// \brief Defines class Evidence, which stores multiple observations of joint states of variables


#ifndef __defined_libdai_evidence_h
#define __defined_libdai_evidence_h


#include <istream>
#include <dai/daialg.h>


namespace dai {


/// Stores a data set consisting of multiple samples, where each sample is the observed joint state of some variables.
/** \note Each sample can describe the joint state of a different set of variables,
 *  in order to be able to deal with missing data.
 *
 *  \author Charles Vaske
 */
class Evidence {
    public:
        /// Stores joint state of a set of variables
        typedef std::map<Var, size_t> Observation;

    private:
        /// Each sample is an observed joint state of some variables
        std::vector<Observation> _samples;

    public:
        /// Default constructor
        Evidence() : _samples() {}

        /// Construct from \a samples
        Evidence( std::vector<Observation> &samples ) : _samples(samples) {}

        /// Read in tabular data from a stream and add the read samples to \c *this.
        /** \param is Input stream in .tab file format, describing joint observations of variables in \a fg
         *  \param fg Factor graph describing the corresponding variables
         *  \see \ref fileformats-evidence
         *  \throw INVALID_EVIDENCE_FILE if the input stream is not valid
         */
        void addEvidenceTabFile( std::istream& is, FactorGraph& fg );

        /// Returns number of stored samples
        size_t nrSamples() const { return _samples.size(); }

    /// \name Iterator interface
    //@{
        /// Iterator over the samples
        typedef std::vector<Observation>::iterator iterator;
        /// Constant iterator over the samples
        typedef std::vector<Observation>::const_iterator const_iterator;

        /// Returns iterator that points to the first sample
        iterator begin() { return _samples.begin(); }
        /// Returns constant iterator that points to the first sample
        const_iterator begin() const { return _samples.begin(); }
        /// Returns iterator that points beyond the last sample
        iterator end() { return _samples.end(); }
        /// Returns constant iterator that points beyond the last sample
        const_iterator end() const { return _samples.end(); }
    //@}

    private:
        /// Read in tabular data from a stream and add the read samples to \c *this.
        void addEvidenceTabFile( std::istream& is, std::map<std::string, Var> &varMap );
};


} // end of namespace dai


#endif
