/*  This file is part of libDAI - http://www.libdai.org/
 *
 *  Copyright (c) 2006-2011, The libDAI authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
 */


#include <iostream>
#include <dai/matlab/matlab.h>
#include "mex.h"
#include <dai/alldai.h>
#include <dai/bp.h>
#include <dai/jtree.h>


using namespace std;
using namespace dai;


/* Input Arguments */

#define PSI_IN          prhs[0]
#define METHOD_IN       prhs[1]
#define OPTS_IN         prhs[2]
#define NR_IN           3
#define NR_IN_OPT       0


/* Output Arguments */

#define LOGZ_OUT        plhs[0]
#define Q_OUT           plhs[1]
#define MD_OUT          plhs[2]
#define QV_OUT          plhs[3]
#define QF_OUT          plhs[4]
#define QMAP_OUT        plhs[5]
#define NR_OUT          3
#define NR_OUT_OPT      3


void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray*prhs[] ) {
    size_t buflen;

    // Check for proper number of arguments
    if( ((nrhs < NR_IN) || (nrhs > NR_IN + NR_IN_OPT)) || ((nlhs < NR_OUT) || (nlhs > NR_OUT + NR_OUT_OPT)) ) {
        mexErrMsgTxt("Usage: [logZ,q,md,qv,qf,qmap] = dai(psi,method,opts)\n\n"
        "\n"
        "INPUT:  psi        = linear cell array containing the factors\n"
        "                     (psi{i} should be a structure with a Member field\n"
        "                     and a P field).\n"
        "        method     = name of the method\n"
        "        opts       = string of options\n"
        "\n"
        "OUTPUT: logZ       = approximation of the logarithm of the partition sum.\n"
        "        q          = linear cell array containing all final beliefs.\n"
        "        md         = maxdiff (final linf-dist between new and old single node beliefs).\n"
        "        qv         = linear cell array containing all variable beliefs.\n"
        "        qf         = linear cell array containing all factor beliefs.\n"
        "        qmap       = linear array containing the MAP state (only for BP,JTree).\n");
    }

    char *method;
    char *opts;


    // Get psi and construct factorgraph
    vector<Factor> factors = mx2Factors(PSI_IN, 0);
    FactorGraph fg(factors);

    // Get method
    buflen = mxGetN( METHOD_IN ) + 1;
    method = (char *)mxCalloc( buflen, sizeof(char) );
    mxGetString( METHOD_IN, method, buflen );

    // Get options string
    buflen = mxGetN( OPTS_IN ) + 1;
    opts = (char *)mxCalloc( buflen, sizeof(char) );
    mxGetString( OPTS_IN, opts, buflen );
    // Convert to options object props
    stringstream ss;
    ss << opts;
    PropertySet props;
    ss >> props;

    // Construct InfAlg object, init and run
    InfAlg *obj = newInfAlg( method, fg, props );
    obj->init();
    obj->run();

    // Save logZ
	double logZ = NAN;
    try {
        logZ = obj->logZ();
    }
    catch( Exception &e ) {
        if( e.getCode() == Exception::NOT_IMPLEMENTED )
            mexWarnMsgTxt("Calculating the log-partition function is not supported by this inference algorithm.");
        else
            throw;
    }

    // Save maxdiff
    double maxdiff = NAN; 
    try {
        maxdiff = obj->maxDiff();
    }
    catch( Exception &e ) {
        if( e.getCode() == Exception::NOT_IMPLEMENTED )
            mexWarnMsgTxt("Calculating the max-differences is not supported by this inference algorithm.");
        else
            throw;
    }

    // Hand over results to MATLAB
    LOGZ_OUT = mxCreateDoubleMatrix(1,1,mxREAL);
    *(mxGetPr(LOGZ_OUT)) = logZ;

    Q_OUT = Factors2mx(obj->beliefs());

    MD_OUT = mxCreateDoubleMatrix(1,1,mxREAL);
    *(mxGetPr(MD_OUT)) = maxdiff;

    if( nlhs >= 4 ) {
        vector<Factor> qv;
        qv.reserve( fg.nrVars() );
        for( size_t i = 0; i < fg.nrVars(); i++ )
            qv.push_back( obj->belief( fg.var(i) ) );
        QV_OUT = Factors2mx( qv );
    }

    if( nlhs >= 5 ) {
        vector<Factor> qf;
        qf.reserve( fg.nrFactors() );
        for( size_t I = 0; I < fg.nrFactors(); I++ )
            qf.push_back( obj->belief( fg.factor(I).vars() ) );
        QF_OUT = Factors2mx( qf );
    }

    if( nlhs >= 6 ) {
        std::vector<std::size_t> map_state;
        bool supported = true;
        try {
            map_state = obj->findMaximum();
        } catch( Exception &e ) {
            if( e.getCode() == Exception::NOT_IMPLEMENTED )
                supported = false;
            else
                throw;
        }
        if( supported ) {
            QMAP_OUT = mxCreateNumericMatrix(map_state.size(), 1, mxUINT32_CLASS, mxREAL);
            uint32_T* qmap_p = reinterpret_cast<uint32_T *>(mxGetPr(QMAP_OUT));
            for (size_t n = 0; n < map_state.size(); ++n)
                qmap_p[n] = map_state[n];
        } else {
            delete obj;
            mexErrMsgTxt("Calculating a MAP state is not supported by this inference algorithm.");
        }
    }

    delete obj;

    return;
}
