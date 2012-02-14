/*  This file is part of libDAI - http://www.libdai.org/
 *
 *  Copyright (c) 2006-2011, The libDAI authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
 */


/** \file
 *  \brief Contains additional doxygen documentation
 *
 *  \idea Adapt (part of the) guidelines in http://www.boost.org/development/requirements.html#Design_and_Programming
 *
 *  \idea Use "gcc -MM" to generate dependencies for targets: http://make.paulandlesley.org/autodep.html
 *
 *  \idea Disentangle structures. In particular, ensure that graphical properties are not
 *  entangled with probabilistic properties. For example, a FactorGraph contains several components:
 *  - a BipartiteGraph
 *  - an array of variable labels
 *  - an array of variable state space sizes
 *  - an array of pointers to factor value vectors
 *  In this way, each factor could be implemented differently, e.g., we could have
 *  some sparse factors, some noisy-OR factors, some dense factors, some arbitrary
 *  precision factors, etcetera.
 *
 *  \idea Use boost::uBLAS framework to deal with matrices, especially, with 2D sparse matrices.
 *  See http://www.boost.org/libs/numeric/ublas/doc/matrix_sparse.htm
 *  However: I read somewhere that boost::uBLAS concentrates more on correct implementation than on performance.
 **/


/** \mainpage Reference manual for libDAI - A free/open source C++ library for Discrete Approximate Inference methods
 *  \author Joris Mooij (with contributions of Frederik Eaton)
 *  \version git HEAD
 *  \date July 12, 2011 - or later
 *
 *  <hr size="1">
 *  \section about About libDAI
 *  libDAI is a free/open source C++ library that provides implementations of
 *  various (approximate) inference methods for discrete graphical models. libDAI
 *  supports arbitrary factor graphs with discrete variables; this includes
 *  discrete Markov Random Fields and Bayesian Networks.
 *
 *  The library is targeted at researchers. To be able to use the library, a
 *  good understanding of graphical models is needed.
 *
 *  The best way to use libDAI is by writing C++ code that invokes the library;
 *  in addition, part of the functionality is accessibly by using the
 *  - command line interface
 *  - (limited) MatLab interface
 *  - (experimental) python interface
 *  - (experimental) octave interface.
 *
 *  libDAI can be used to implement novel (approximate) inference algorithms
 *  and to easily compare the accuracy and performance with existing algorithms
 *  that have been implemented already.
 *
 *  A solver using libDAI was amongst the three winners of the UAI 2010 Approximate
 *  Inference Challenge (see http://www.cs.huji.ac.il/project/UAI10/ for more 
 *  information). The full source code is provided as part of the library.
 *
 *  \section features Features
 *  Currently, libDAI supports the following (approximate) inference methods:
 *  - Exact inference by brute force enumeration;
 *  - Exact inference by junction-tree methods;
 *  - Mean Field;
 *  - Loopy Belief Propagation [\ref KFL01];
 *  - Fractional Belief Propagation [\ref WiH03];
 *  - Tree-Reweighted Belief Propagation [\ref WJW03];
 *  - Tree Expectation Propagation [\ref MiQ04];
 *  - Generalized Belief Propagation [\ref YFW05];
 *  - Double-loop GBP [\ref HAK03];
 *  - Various variants of Loop Corrected Belief Propagation
 *    [\ref MoK07, \ref MoR05];
 *  - Gibbs sampler;
 *  - Conditioned Belief Propagation [\ref EaG09];
 *  - Decimation algorithm.
 *
 *  These inference methods can be used to calculate partition sums, marginals
 *  over subsets of variables, and MAP states (the joint state of variables that
 *  has maximum probability).
 *
 *  In addition, libDAI supports parameter learning of conditional probability
 *  tables by Expectation Maximization.
 *
 *  \section limitations Limitations
 *  libDAI is not intended to be a complete package for approximate inference.
 *  Instead, it should be considered as an "inference engine", providing
 *  various inference methods. In particular, it contains no GUI, currently
 *  only supports its own file format for input and output (although support
 *  for standard file formats may be added later), and provides very limited
 *  visualization functionalities. The only learning method supported currently
 *  is Expectation Maximization (or Maximum Likelihood if no data is missing)
 *  for learning factor parameters.
 *
 *  \section rationale Rationale
 *
 *  In my opinion, the lack of open source "reference" implementations hampers
 *  progress in research on approximate inference. Methods differ widely in terms
 *  of quality and performance characteristics, which also depend in different
 *  ways on various properties of the graphical models. Finding the best
 *  approximate inference method for a particular application therefore often
 *  requires empirical comparisons. However, implementing and debugging these
 *  methods takes a lot of time which could otherwise be spent on research. I hope
 *  that this code will aid researchers to be able to easily compare various
 *  (existing as well as new) approximate inference methods, in this way
 *  accelerating research and stimulating real-world applications of approximate
 *  inference.
 *
 *  \section language Language
 *  Because libDAI is implemented in C++, it is very fast compared with
 *  implementations in MatLab (a factor 1000 faster is not uncommon).
 *  libDAI does provide a (limited) MatLab interface for easy integration with MatLab.
 *  It also provides a command line interface and experimental python and octave 
 *  interfaces (thanks to Patrick Pletscher).
 *
 *  \section compatibility Compatibility
 *  
 *  The code has been developed under Debian GNU/Linux with the GCC compiler suite.
 *  libDAI compiles successfully with g++ versions 3.4 up to 4.6.
 *
 *  libDAI has also been successfully compiled with MS Visual Studio 2008 under Windows
 *  (but not all build targets are supported yet) and with Cygwin under Windows.
 *
 *  Finally, libDAI has been compiled successfully on MacOS X.
 *
 *  \section download Downloading libDAI
 *  The libDAI sources and documentation can be downloaded from the libDAI website:
 *  http://www.libdai.org.
 *
 *  \section support Mailing list
 *  The Google group "libDAI" (http://groups.google.com/group/libdai)
 *  can be used for getting support and discussing development issues.
 */


/** \page license License
 *  <hr size="1">
 *  \section license-license License
 *
 *  libDAI is free software; you can redistribute it and/or modify it under the
 *  terms of the BSD 2-clause license (also known as the FreeBSD license), which
 *  can be found in the accompanying LICENSE file.
 *
 *  [Note: up to and including version 0.2.7, libDAI was licensed under the GNU 
 *  General Public License (GPL) version 2 or higher.]
 *
 *  <hr size="1">
 *  \section license-freebsd libDAI license (similar to FreeBSD license)
 * 
 *  \verbinclude LICENSE
 */


/** \page citations Citing libDAI
 *  <hr size="1">
 *  \section citations-citations Citing libDAI
 *
 *  If you write a scientific paper describing research that made substantive use
 *  of this library, please cite the following paper describing libDAI:\n
 *
 *  Joris M. Mooij;\n
 *  libDAI: A free & open source C++ library for Discrete Approximate Inference in graphical models;\n
 *  Journal of Machine Learning Research, 11(Aug):2169-2173, 2010.\n
 *
 *  In BiBTeX format (for your convenience):\n
 *  
 *  <pre>
 *  \@article{Mooij_libDAI_10,
 *    author    = {Joris M. Mooij},
 *    title     = {lib{DAI}: A Free and Open Source {C++} Library for Discrete Approximate Inference in Graphical Models},
 *    journal   = {Journal of Machine Learning Research},
 *    year      = 2010,
 *    month     = Aug,
 *    volume    = 11,
 *    pages     = {2169-2173},
 *    url       = "http://www.jmlr.org/papers/volume11/mooij10a/mooij10a.pdf"
 *  }</pre>
 *
 *  Moreover, as a personal note, I would appreciate it to be informed about any
 *  publications using libDAI at joris dot mooij at libdai dot org.
 */


/** \page authors Authors
 *  \section authors-authors People who contributed to libDAI
 *
 *  \verbinclude AUTHORS
 */


/** \page build Building libDAI
 *  <hr size="1">
 *  \section build-unix Building libDAI under UNIX variants (Linux / Cygwin / Mac OS X)
 *
 *  \subsection build-unix-preparations Preparations
 *
 *  You need:
 *    - a recent version of gcc (at least version 3.4)
 *    - GNU make
 *    - recent boost C++ libraries (at least version 1.37; however,
 *      version 1.37 shipped with Ubuntu 9.04 is known not to work)
 *    - GMP library (or the Windows port called MPIR)
 *    - doxygen (only for building the documentation)
 *    - graphviz (only for using some of the libDAI command line utilities)
 *    - CImg library (only for building the image segmentation example)
 * 
 *  On Debian/Ubuntu, you can easily install the required packages with a single command:
 *  <pre>  apt-get install g++ make doxygen graphviz libboost-dev libboost-graph-dev libboost-program-options-dev libboost-test-dev libgmp-dev cimg-dev</pre>
 *  (root permissions needed).
 *
 *  On Mac OS X (10.4 is known to work), these packages can be installed easily via MacPorts.
 *  If MacPorts is not already installed, install it according to the instructions at http://www.macports.org/.
 *  Then, a simple 
 *    <pre>  sudo port install gmake boost gmp doxygen graphviz</pre>
 *  should be enough to install everything that is needed.
 *  
 *  On Cygwin, the prebuilt Cygwin package boost-1.33.1-x is known not to work.
 *  You can however obtain the latest boost version (you need at least 1.37.0)
 *  from http://www.boost.org/ and build it as described in the next subsection.
 *
 *  \subsubsection build-unix-boost Building boost under Cygwin
 *
 *  - Download the latest boost libraries from http://www.boost.org
 *  - Build the required boost libraries using:
 *    <pre>
 *    ./bootstrap.sh --with-libraries=program_options,math,graph,test --prefix=/boost_root/
 *    ./bjam</pre>
 *  - In order to use dynamic linking, the boost .dll's should be somewhere in the path. 
 *    This can be achieved by a command like:
 *    <pre>
 *    export PATH=$PATH:/boost_root/stage/lib</pre>
 *
 *
 *  \subsection build-unix-libdai Building libDAI
 *
 *  To build the libDAI source, first copy a template Makefile.* to Makefile.conf
 *  (for example, copy Makefile.LINUX to Makefile.conf if you use GNU/Linux). 
 *  Then, edit the Makefile.conf template to adapt it to your local setup. In case
 *  you want to use Boost libraries which are installed in non-standard locations,
 *  you have to tell the compiler and linker about their locations (using the
 *  -I, -L flags for GCC; also you may need to set the LD_LIBRARY_PATH environment 
 *  variable correctly before running libDAI binaries). Platform independent build
 *  options can be set in Makefile.ALL. Finally, run
 *  <pre>  make</pre>
 *  The build includes a regression test, which may take a while to complete.
 *
 *  If the build is successful, you can test the example program:
 *  <pre>  examples/example tests/alarm.fg</pre>
 *  or the more extensive test program:
 *  <pre>  tests/testdai --aliases tests/aliases.conf --filename tests/alarm.fg --methods JTREE_HUGIN BP_SEQMAX</pre>
 *
 *
 *  <hr size="1">
 *  \section build-windows Building libDAI under Windows
 *
 *  \subsection build-windows-preparations Preparations
 *
 *  You need:
 *  - A recent version of MicroSoft Visual Studio (2008 is known to work)
 *  - recent boost C++ libraries (version 1.37 or higher)
 *  - GMP or MPIR library
 *  - GNU make (can be obtained from http://gnuwin32.sourceforge.net)
 *  - CImg library (only for building the image segmentation example)
 *
 *  For the regression test, you need:
 *  - GNU diff, GNU sed (can be obtained from http://gnuwin32.sourceforge.net)
 *
 *  \subsubsection build-windows-boost Building boost under Windows
 *
 *  Because building boost under Windows is tricky, I provide some guidance here.
 *
 *  - Download the boost zip file from http://www.boost.org/users/download
 *    and unpack it somewhere.
 *  - Download the bjam executable from http://www.boost.org/users/download
 *    and unpack it somewhere else.
 *  - Download Boost.Build (v2) from http://www.boost.org/docs/tools/build/index.html
 *    and unpack it yet somewhere else.
 *  - Edit the file \c boost-build.jam in the main boost directory to change the
 *    \c BOOST_BUILD directory to the place where you put Boost.Build (use UNIX
 *    / instead of Windows \ in pathnames).
 *  - Copy the \c bjam.exe executable into the main boost directory.
 *    Now if you issue <tt>"bjam --version"</tt> you should get a version and no errors.
 *    Issueing <tt>"bjam --show-libraries"</tt> will show the libraries that will be built.
 *  - The following command builds the boost libraries that are relevant for libDAI:
 *    <pre>
 *    bjam --with-graph --with-math --with-program_options --with-test link=static runtime-link=shared</pre>
 *
 *  \subsection build-windows-gmp Building GMP or MPIR under Windows
 *  
 *  Information about how to build GPR or MPIR under Windows can be found on the internet.
 *  The user has to update Makefile.WINDOWS in order to link with the GPR/MPIR libraries.
 *
 *  \subsection build-windows-libdai Building libDAI
 *
 *  To build the source, copy Makefile.WINDOWS to Makefile.conf. Then, edit 
 *  Makefile.conf to adapt it to your local setup. Platform independent 
 *  build options can be set in Makefile.ALL. Finally, run (from the command line)
 *  <pre>  make</pre>
 *  The build includes a regression test, which may take a while to complete.
 *
 *  If the build is successful, you can test the example program:
 *  <pre>  examples\\example tests\\alarm.fg</pre>
 *  or the more extensive test program:
 *  <pre>  tests\\testdai --aliases tests\\aliases.conf --filename tests\\alarm.fg --methods JTREE_HUGIN BP_SEQMAX</pre>
 *
 *
 *  <hr size="1">
 *  \section build-matlab Building the libDAI MatLab interface
 *
 *  You need:
 *  - MatLab
 *  - The platform-dependent requirements described above
 *
 *  First, you need to build the libDAI source as described above for your
 *  platform. By default, the MatLab interface is disabled, so before compiling the
 *  source, you have to enable it in Makefile.ALL by setting
 *  <pre>  WITH_MATLAB=true</pre>
 *  Also, you have to configure the MatLab-specific parts of
 *  Makefile.conf to match your system (in particular, the Makefile variables ME,
 *  MATLABDIR and MEX). The MEX file extension depends on your platform; for a
 *  64-bit linux x86_64 system this would be "ME=.mexa64", for a 32-bit linux x86
 *  system "ME=.mexglx". If you are unsure about your MEX file
 *  extension: it needs to be the same as what the MatLab command "mexext" returns.
 *  The required MEX files are built by issuing
 *  <pre>  make</pre>
 *  from the command line. The MatLab interface is much less powerful than using
 *  libDAI from C++. There are two reasons for this: (i) it is boring to write MEX
 *  files; (ii) the large performance penalty paid when large data structures (like
 *  factor graphs) have to be converted between their native C++ data structure to
 *  something that MatLab understands.
 *
 *  A simple example of how to use the MatLab interface is the following (entered
 *  at the MatLab prompt), which performs exact inference by the junction tree
 *  algorithm and approximate inference by belief propagation on the ALARM network:
 *  <pre>  cd path_to_libdai/matlab
 *  [psi] = dai_readfg ('../tests/alarm.fg');
 *  [logZ,q,md,qv,qf] = dai (psi, 'JTREE', '[updates=HUGIN,verbose=0]')
 *  [logZ,q,md,qv,qf] = dai (psi, 'BP', '[updates=SEQMAX,tol=1e-9,maxiter=10000,logdomain=0]')</pre>
 *  where "path_to_libdai" has to be replaced with the directory in which libDAI
 *  was installed. For other algorithms and some default parameters, see the file
 *  tests/aliases.conf.
 *
 *  <hr size="1">
 *  \section build-doxygen Building the documentation
 *
 *  Install doxygen, graphviz and a TeX distribution and use
 *  <pre>  make doc</pre>
 *  to build the documentation. If the documentation is not clear enough, feel free 
 *  to send me an email (or even better, to improve the documentation and send a patch!).
 *  The documentation can also be browsed online at http://www.libdai.org.
 */


/** \page changelog Change Log
 *  \verbinclude ChangeLog
 */


/** \page terminology Terminology and conventions
 *
 *  \section terminology-graphicalmodels Graphical models
 *
 *  Commonly used graphical models are Bayesian networks and Markov random fields.
 *  In libDAI, both types of graphical models are represented by a slightly more 
 *  general type of graphical model: a factor graph [\ref KFL01].
 *
 *  An example of a Bayesian network is: 
 *  \dot
 *  digraph bayesnet {
 *    size="1,1";
 *    x0 [label="0"];
 *    x1 [label="1"];
 *    x2 [label="2"];
 *    x3 [label="3"];
 *    x4 [label="4"];
 *    x0 -> x1;
 *    x0 -> x2;
 *    x1 -> x3;
 *    x1 -> x4;
 *    x2 -> x4;
 *  }
 *  \enddot
 *  The probability distribution of a Bayesian network factorizes as:
 *  \f[ P(\mathbf{x}) = \prod_{i\in\mathcal{V}} P(x_i \,|\, x_{\mathrm{pa}(i)}) \f]
 *  where \f$\mathrm{pa}(i)\f$ are the parents of node \a i in a DAG.
 *
 *  The same probability distribution can be represented as a Markov random field:
 *  \dot
 *  graph mrf {
 *    size="1.5,1.5";
 *    x0 [label="0"];
 *    x1 [label="1"];
 *    x2 [label="2"];
 *    x3 [label="3"];
 *    x4 [label="4"];
 *    x0 -- x1;
 *    x0 -- x2;
 *    x1 -- x2;
 *    x1 -- x3;
 *    x1 -- x4;
 *    x2 -- x4;
 *  }
 *  \enddot
 *
 *  The probability distribution of a Markov random field factorizes as:
 *  \f[ P(\mathbf{x}) = \frac{1}{Z} \prod_{C\in\mathcal{C}} \psi_C(x_C) \f]
 *  where \f$ \mathcal{C} \f$ are the cliques of an undirected graph, 
 *  \f$ \psi_C(x_C) \f$ are "potentials" or "compatibility functions", and
 *  \f$ Z \f$ is the partition sum which properly normalizes the probability
 *  distribution.
 *
 *  Finally, the same probability distribution can be represented as a factor graph:
 *  \dot
 *  graph factorgraph {
 *    size="1.8,1";
 *    x0 [label="0"];
 *    x1 [label="1"];
 *    x2 [label="2"];
 *    x3 [label="3"];
 *    x4 [label="4"];
 *    f01 [shape="box",label=""];
 *    f02 [shape="box",label=""];
 *    f13 [shape="box",label=""];
 *    f124 [shape="box",label=""];
 *    x0 -- f01;
 *    x1 -- f01;
 *    x0 -- f02;
 *    x2 -- f02;
 *    x1 -- f13;
 *    x3 -- f13;
 *    x1 -- f124;
 *    x2 -- f124;
 *    x4 -- f124;
 *  }
 *  \enddot
 *
 *  The probability distribution of a factor graph factorizes as:
 *  \f[ P(\mathbf{x}) = \frac{1}{Z} \prod_{I\in \mathcal{F}} f_I(x_I) \f]
 *  where \f$ \mathcal{F} \f$ are the factor nodes of a factor graph (a 
 *  bipartite graph consisting of variable nodes and factor nodes), 
 *  \f$ f_I(x_I) \f$ are the factors, and \f$ Z \f$ is the partition sum
 *  which properly normalizes the probability distribution.
 *
 *  Looking at the expressions for the joint probability distributions,
 *  it is obvious that Bayesian networks and Markov random fields can 
 *  both be easily represented as factor graphs. Factor graphs most
 *  naturally express the factorization structure of a probability
 *  distribution, and hence are a convenient representation for approximate
 *  inference algorithms, which all try to exploit this factorization.
 *  This is why libDAI uses a factor graph as representation of a 
 *  graphical model, implemented in the dai::FactorGraph class.
 *
 *  \section terminology-inference Inference tasks
 *
 *  Given a factor graph, specified by the variable nodes \f$\{x_i\}_{i\in\mathcal{V}}\f$
 *  the factor nodes \f$ \mathcal{F} \f$, the graph structure, and the factors
 *  \f$\{f_I(x_I)\}_{I\in\mathcal{F}}\f$, the following tasks are important:
 *
 *  - Calculating the partition sum:
 *    \f[ Z = \sum_{\mathbf{x}_{\mathcal{V}}} \prod_{I \in \mathcal{F}} f_I(x_I) \f]
 *  - Calculating the marginal distribution of a subset of variables
 *    \f$\{x_i\}_{i\in A}\f$: 
 *    \f[ P(\mathbf{x}_{A}) = \frac{1}{Z} \sum_{\mathbf{x}_{\mathcal{V}\setminus A}} \prod_{I \in \mathcal{F}} f_I(x_I) \f]
 *  - Calculating the MAP state which has the maximum probability mass:
 *    \f[ \mathrm{argmax}_{\mathbf{x}}\,\prod_{I\in\mathcal{F}} f_I(x_I) \f]
 *
 *  libDAI offers several inference algorithms, which solve (a subset of) these tasks either 
 *  approximately or exactly, for factor graphs with discrete variables. The following
 *  algorithms are implemented:
 *  
 *  Exact inference:
 *  - Brute force enumeration: dai::ExactInf
 *  - Junction-tree method: dai::JTree
 *
 *  Approximate inference:
 *  - Mean Field: dai::MF
 *  - (Loopy) Belief Propagation: dai::BP [\ref KFL01]
 *  - Fractional Belief Propagation: dai::FBP [\ref WiH03]
 *  - Tree-Reweighted Belief Propagation: dai::TRWBP [\ref WJW03]
 *  - Tree Expectation Propagation: dai::TreeEP [\ref MiQ04]
 *  - Generalized Belief Propagation: dai::HAK [\ref YFW05]
 *  - Double-loop GBP: dai::HAK [\ref HAK03]
 *  - Loop Corrected Belief Propagation: dai::MR [\ref MoR05] and dai::LC [\ref MoK07]
 *  - Gibbs sampling: dai::Gibbs
 *  - Conditioned Belief Propagation: dai::CBP [\ref EaG09]
 *  - Decimation algorithm: dai::DecMAP
 *
 *  Not all inference tasks are implemented by each method: calculating MAP states
 *  is only possible with dai::JTree, dai::BP and dai::DECMAP; calculating partition sums is
 *  not possible with dai::MR, dai::LC and dai::Gibbs.
 *
 *  \section terminology-learning Parameter learning
 *
 *  In addition, libDAI supports parameter learning of conditional probability
 *  tables by Expectation Maximization (or Maximum Likelihood, if there is no
 *  missing data). This is implemented in dai::EMAlg.
 *  
 *  \section terminology-variables-states Variables and states
 *
 *  Linear states are a concept that is used often in libDAI, for example for storing
 *  and accessing factors, which are functions mapping from states of a set of variables
 *  to the real numbers. Internally, a factor is stored as an array, and the array index
 *  of an entry corresponds with the linear state of the set of variables. Below we will
 *  define variables, states and linear states of (sets of) variables.
 *
 *  \subsection terminology-variables Variables
 *
 *  Each (random) \a variable has a unique identifier, its \a label (which has
 *  a non-negative integer value). If two variables have the same
 *  label, they are considered as identical. A variable can take on a finite
 *  number of different values or \a states.
 *  
 *  We use the following notational conventions. The discrete
 *  random variable with label \f$l\f$ is denoted as \f$x_l\f$, and the number
 *  of possible values of this variable as \f$S_{x_l}\f$ or simply \f$S_l\f$. 
 *  The set of possible values of variable \f$x_l\f$ is denoted 
 *  \f$X_l := \{0,1,\dots,S_l-1\}\f$ and called its \a state \a space.
 *
 *  \subsection terminology-variable-sets Sets of variables and the canonical ordering
 *  
 *  Let \f$A := \{x_{l_1},x_{l_2},\dots,x_{l_n}\}\f$ be a set of variables. 
 *
 *  The \a canonical \a ordering of the variables in \a A is induced by their labels.
 *  That is: if \f$l_1 < l_2\f$, then \f$x_{l_1}\f$ occurs before \f$x_{l_2}\f$ in the
 *  canonical ordering. Below, we will assume that \f$(l_i)_{i=1}^n\f$ is
 *  ordered according to the canonical ordering, i.e., \f$l_1 < l_2 < \dots < l_n\f$.
 *
 *  \subsection terminology-variable-states States and linear states of sets of variables
 *
 *  A \a state of the variables in \a A refers to a joint assignment of the 
 *  variables, or in other words, to an element of the Cartesian product 
 *  \f$ \prod_{i=1}^n X_{l_i}\f$ of the state spaces of the variables in \a A.
 *  Note that a state can also be interpreted as a mapping from variables (or
 *  variable labels) to the natural numbers, which assigns to a variable (or its
 *  label) the corresponding state of the variable.
 *
 *  A state of \a n variables can be represented as an n-tuple of 
 *  non-negative integers: \f$(s_1,s_2,\dots,s_n)\f$ corresponds to the
 *  joint assignment \f$x_{l_1} = s_1, \dots, x_{l_n} = s_n\f$.
 *  Alternatively, a state can be represented compactly as one non-negative integer;
 *  this representation is called a \a linear \a state. The linear state
 *  \a s corresponding to the state \f$(s_1,s_2,\dots,s_n)\f$ would be:
 *  \f[
 *    s := \sum_{i=1}^n s_i \prod_{j=1}^{i-1} S_{l_j} 
 *       = s_1 + s_2 S_{l_1} + s_3 S_{l_1} S_{l_2} + \dots + s_n S_{l_1} \cdots S_{l_{n-1}}.
 *  \f]
 *
 *  Vice versa, given a linear state \a s for the variables \a A, the
 *  corresponding state \f$s_i\f$ of the \a i 'th variable \f$x_{l_i}\f$ (according to
 *  the canonical ordering of the variables in \a A) is given by
 *  \f[
 *    s_i = \left\lfloor\frac{s \mbox { mod } \prod_{j=1}^i S_{l_j}}{\prod_{j=1}^{i-1} S_{l_j}}\right\rfloor.
 *  \f]
 *
 *  Finally, the \a number \a of \a states of the set of variables \a A is simply the 
 *  number of different joint assignments of the variables, that is, \f$\prod_{i=1}^n S_{l_i}\f$.
 */


/** \page fileformats libDAI file formats
 *
 *  \section fileformats-factorgraph Factor graph (.fg) file format
 *
 *  This section describes the .fg file format used in libDAI to store factor graphs.
 *  Markov Random Fields are special cases of factor graphs, as are Bayesian
 *  networks. A factor graph can be specified as follows: for each factor, one has
 *  to specify which variables occur in the factor, what their respective
 *  cardinalities (i.e., number of possible values) are, and a table listing all
 *  the values of that factor for all possible configurations of these variables.
 *
 *  A .fg file is not much more than that. It starts with a line containing the
 *  number of factors in that graph, followed by an empty line. Then all factors
 *  are specified, using one block for each factor, where the blocks are seperated 
 *  by empty lines. Each variable occurring in the factor graph has a unique
 *  identifier, its label (which should be a nonnegative integer). Comment lines
 *  which start with # are ignored.
 *
 *  \subsection fileformats-factorgraph-factor Factor block format
 *
 *  Each block describing a factor starts with a line containing the number of 
 *  variables in that factor. The second line contains the labels of these 
 *  variables, seperated by spaces (labels are nonnegative integers and to avoid 
 *  confusion, it is suggested to start counting at 0). The third line contains 
 *  the number of possible values of each of these variables, also seperated by 
 *  spaces. Note that there is some redundancy here, since if a variable appears 
 *  in more than one factor, the cardinality of that variable appears several 
 *  times in the .fg file; obviously, these cardinalities should be consistent.
 *  The fourth line contains the number of nonzero entries 
 *  in the factor table. The rest of the lines contain these nonzero entries; 
 *  each line consists of a table index, followed by white-space, followed by the 
 *  value corresponding to that table index. The most difficult part is getting 
 *  the indexing right. The convention that is used is that the left-most 
 *  variables cycle through their values the fastest (similar to MatLab indexing 
 *  of multidimensional arrays). 
 *
 *  \subsubsection fileformats-factorgraph-factor-example Example
 *
 *  An example block describing one factor is:
 *
 *  <pre>
 *  3
 *  4 8 7
 *  3 2 2
 *  11
 *  0 0.1
 *  1 3.5
 *  2 2.8
 *  3 6.3
 *  4 8.4
 *  6 7.4
 *  7 2.4
 *  8 8.9
 *  9 1.3
 *  10 1.6
 *  11 2.6
 *  </pre>
 *
 *  which corresponds to the following factor:
 *
 *  \f[
 *  \begin{array}{ccc|c}
 *  x_4 & x_8 & x_7 & \mbox{value}\\
 *  \hline
 *   0 & 0 & 0  &  0.1\\
 *   1 & 0 & 0  &  3.5\\
 *   2 & 0 & 0  &  2.8\\
 *   0 & 1 & 0  &  6.3\\
 *   1 & 1 & 0  &  8.4\\
 *   2 & 1 & 0  &  0.0\\
 *   0 & 0 & 1  &  7.4\\
 *   1 & 0 & 1  &  2.4\\
 *   2 & 0 & 1  &  8.9\\
 *   0 & 1 & 1  &  1.3\\
 *   1 & 1 & 1  &  1.6\\
 *   2 & 1 & 1  &  2.6
 *  \end{array}
 *  \f]
 *
 *  Note that the value of \f$x_4\f$ changes fastest, followed by that of \f$x_8\f$, and \f$x_7\f$
 *  varies the slowest, corresponding to the second line of the block ("4 8 7").
 *  Further, \f$x_4\f$ can take on three values, and \f$x_8\f$ and \f$x_7\f$ each have two possible
 *  values, as described in the third line of the block ("3 2 2"). The table
 *  contains 11 non-zero entries (all except for the fifth entry). Note that the
 *  eleventh and twelveth entries are interchanged.
 *
 *  A final note: the internal representation in libDAI of the factor above is
 *  different, because the variables are ordered according to their indices
 *  (i.e., the ordering would be \f$x_4 x_7 x_8\f$) and the values of the table are
 *  stored accordingly, with the variable having the smallest index changing
 *  fastest:
 *
 *  \f[
 *  \begin{array}{ccc|c}
 *  x_4 & x_7 & x_8 & \mbox{value}\\
 *  \hline
 *   0 & 0 & 0  &  0.1\\
 *   1 & 0 & 0  &  3.5\\
 *   2 & 0 & 0  &  2.8\\
 *   0 & 1 & 0  &  7.4\\
 *   1 & 1 & 0  &  2.4\\
 *   2 & 1 & 0  &  8.9\\
 *   0 & 0 & 1  &  6.3\\
 *   1 & 0 & 1  &  8.4\\
 *   2 & 0 & 1  &  0.0\\
 *   0 & 1 & 1  &  1.3\\
 *   1 & 1 & 1  &  1.6\\
 *   2 & 1 & 1  &  2.6
 *  \end{array}
 *  \f]
 *
 *
 *  \section fileformats-evidence Evidence (.tab) file format
 *
 *  This section describes the .tab fileformat used in libDAI to store "evidence",
 *  i.e., a data set consisting of multiple samples, where each sample is the 
 *  observed joint state of some variables.
 *
 *  A .tab file is a tabular data file, consisting of a header line, followed by
 *  an empty line, followed by the data points, with one line for each data point.
 *  Each line (apart from the empty one) should have the same number of columns,
 *  where columns are separated by one tab character. Each column corresponds to 
 *  a variable. The header line consists of the variable labels (corresponding to 
 *  dai::Var::label()). The other lines are observed joint states of the variables, i.e.,
 *  each line corresponds to a joint observation of the variables, and each column
 *  of a line contains the state of the variable associated with that column.
 *  Missing data is handled simply by having two consecutive tab characters, 
 *  without any characters in between.
 *
 *  \subsection fileformats-evidence-example Example
 *
 *  <pre>
 *  1       3       2
 *
 *  0       0       1
 *  1       0       1
 *  1               1
 *  </pre>
 *
 *  This would correspond to a data set consisting of three observations concerning
 *  the variables with labels 1, 3 and 2; the first observation being
 *  \f$x_1 = 0, x_3 = 0, x_2 = 1\f$, the second observation being
 *  \f$x_1 = 1, x_3 = 0, x_2 = 1\f$, and the third observation being
 *  \f$x_1 = 1, x_2 = 1\f$ (where the state of \f$x_3\f$ is missing).
 *
 *  \section fileformats-emalg Expectation Maximization (.em) file format
 *
 *  This section describes the file format of .em files, which are used
 *  to specify a particular EM algorithm. The .em files are complementary
 *  to .fg files; in other words, an .em file without a corresponding .fg 
 *  file is useless. Furthermore, one also needs a corresponding .tab file
 *  containing the data used for parameter learning.
 *
 *  An .em file starts with a line specifying the number of maximization steps,
 *  followed by an empty line. Then, each maximization step is described in a
 *  block, which should satisfy the format described in the next subsection.
 *
 *  \subsection fileformats-emalg-maximizationstep Maximization Step block format
 *
 *  A maximization step block of an .em file starts with a single line
 *  describing the number of shared parameters blocks that will follow.
 *  Then, each shared parameters block follows, in the format described in
 *  the next subsection.
 *
 *  \subsection fileformats-emalg-sharedparameters Shared parameters block format
 *
 *  A shared parameters block of an .em file starts with a single line
 *  consisting of the name of a ParameterEstimation subclass
 *  and its parameters in the format of a PropertySet. For example:
 *  <pre>  CondProbEstimation [target_dim=2,total_dim=4,pseudo_count=1]</pre>
 *  The next line contains the number of factors that share their parameters.
 *  Then, each of these factors is specified on separate lines (possibly 
 *  seperated by empty lines), where each line consists of several fields
 *  seperated by a space or a tab character. The first field contains 
 *  the index of the factor in the factor graph. The following fields should
 *  contain the variable labels of the variables on which that factor depends, 
 *  in a specific ordering. This ordering can be different from the canonical 
 *  ordering of the variables used internally in libDAI (which would be sorted 
 *  ascendingly according to the variable labels). The ordering of the variables
 *  specifies the implicit ordering of the shared parameters: when iterating
 *  over all shared parameters, the corresponding index of the first variable
 *  changes fastest (in the inner loop), and the corresponding index of the
 *  last variable changes slowest (in the outer loop). By choosing the right
 *  ordering, it is possible to let different factors (depending on different
 *  variables) share parameters in parameter learning using EM. This convention
 *  is similar to the convention used in factor blocks in a factor graph .fg 
 *  file (see \ref fileformats-factorgraph-factor).
 *
 *  \section fileformats-aliases Aliases file format
 *
 *  An aliases file is basically a list of "macros" and the strings that they
 *  should be substituted with.
 *  
 *  Each line of the aliases file can be either empty, contain a comment 
 *  (if the first character is a '#') or contain an alias. In the latter case, 
 *  the line should contain a colon; the part before the colon contains the 
 *  name of the alias, the part after the colon the string that it should be 
 *  substituted with. Any whitespace before and after the colon is ignored.
 *
 *  For example, the following line would define the alias \c BP_SEQFIX
 *  as a shorthand for "BP[updates=SEQFIX,tol=1e-9,maxiter=10000,logdomain=0]":
 *  <pre>
 *  BP_SEQFIX:  BP[updates=SEQFIX,tol=1e-9,maxiter=10000,logdomain=0]
 *  </pre>
 *
 *  Aliases files can be used to store default options for algorithms.
 */

/** \page bibliography Bibliography
 *  \anchor EaG09 \ref EaG09
 *  F. Eaton and Z. Ghahramani (2009):
 *  "Choosing a Variable to Clamp",
 *  <em>Proceedings of the Twelfth International Conference on Artificial Intelligence and Statistics (AISTATS 2009)</em> 5:145-152,
 *  http://jmlr.csail.mit.edu/proceedings/papers/v5/eaton09a/eaton09a.pdf
 *
 *  \anchor EMK06 \ref EMK06
 *  G. Elidan and I. McGraw and D. Koller (2006):
 *  "Residual Belief Propagation: Informed Scheduling for Asynchronous Message Passing",
 *  <em>Proceedings of the 22nd Annual Conference on Uncertainty in Artificial Intelligence (UAI-06)</em>,
 *  http://uai.sis.pitt.edu/papers/06/UAI2006_0091.pdf
 *
 *  \anchor HAK03 \ref HAK03
 *  T. Heskes and C. A. Albers and H. J. Kappen (2003):
 *  "Approximate Inference and Constrained Optimization",
 *  <em>Proceedings of the 19th Annual Conference on Uncertainty in Artificial Intelligence (UAI-03)</em> pp. 313-320,
 *  http://www.snn.ru.nl/reports/Heskes.uai2003.ps.gz
 *
 *  \anchor KFL01 \ref KFL01
 *  F. R. Kschischang and B. J. Frey and H.-A. Loeliger (2001):
 *  "Factor Graphs and the Sum-Product Algorithm",
 *  <em>IEEE Transactions on Information Theory</em> 47(2):498-519,
 *  http://ieeexplore.ieee.org/xpl/freeabs_all.jsp?arnumber=910572
 *
 *  \anchor KoF09 \ref KoF09
 *  D. Koller and N. Friedman (2009):
 *  <em>Probabilistic Graphical Models - Principles and Techniques</em>,
 *  The MIT Press, Cambridge, Massachusetts, London, England.

 *  \anchor Min05 \ref Min05
 *  T. Minka (2005):
 *  "Divergence measures and message passing",
 *  <em>MicroSoft Research Technical Report</em> MSR-TR-2005-173,
 *  http://research.microsoft.com/en-us/um/people/minka/papers/message-passing/minka-divergence.pdf
 *
 *  \anchor MiQ04 \ref MiQ04
 *  T. Minka and Y. Qi (2004):
 *  "Tree-structured Approximations by Expectation Propagation",
 *  <em>Advances in Neural Information Processing Systems</em> (NIPS) 16,
 *  http://books.nips.cc/papers/files/nips16/NIPS2003_AA25.pdf
 *
 *  \anchor MoK07 \ref MoK07
 *  J. M. Mooij and H. J. Kappen (2007):
 *  "Loop Corrections for Approximate Inference on Factor Graphs",
 *  <em>Journal of Machine Learning Research</em> 8:1113-1143,
 *  http://www.jmlr.org/papers/volume8/mooij07a/mooij07a.pdf
 *
 *  \anchor MoK07b \ref MoK07b
 *  J. M. Mooij and H. J. Kappen (2007):
 *  "Sufficient Conditions for Convergence of the Sum-Product Algorithm",
 *  <em>IEEE Transactions on Information Theory</em> 53(12):4422-4437,
 *  http://ieeexplore.ieee.org/xpl/freeabs_all.jsp?arnumber=4385778
 *
 *  \anchor Moo08 \ref Moo08
 *  J. M. Mooij (2008):
 *  "Understanding and Improving Belief Propagation",
 *  <em>Ph.D. Thesis</em> Radboud University Nijmegen
 *  http://webdoc.ubn.ru.nl/mono/m/mooij_j/undeanimb.pdf
 *
 *  \anchor MoR05 \ref MoR05
 *  A. Montanari and T. Rizzo (2005):
 *  "How to Compute Loop Corrections to the Bethe Approximation",
 *  <em>Journal of Statistical Mechanics: Theory and Experiment</em> 2005(10)-P10011,
 *  http://stacks.iop.org/1742-5468/2005/P10011
 *
 *  \anchor StW99 \ref StW99
 *  A. Steger and N. C. Wormald (1999):
 *  "Generating Random Regular Graphs Quickly",
 *  <em>Combinatorics, Probability and Computing</em> Vol 8, Issue 4, pp. 377-396,
 *  http://www.math.uwaterloo.ca/~nwormald/papers/randgen.pdf
 *
 *  \anchor WiH03 \ref WiH03
 *  W. Wiegerinck and T. Heskes (2003):
 *  "Fractional Belief Propagation",
 *  <em>Advances in Neural Information Processing Systems</em> (NIPS) 15, pp. 438-445,
 *  http://books.nips.cc/papers/files/nips15/LT16.pdf
 *
 *  \anchor WJW03 \ref WJW03
 *  M. J. Wainwright, T. S. Jaakkola and A. S. Willsky (2003):
 *  "Tree-reweighted belief propagation algorithms and approximate ML estimation by pseudo-moment matching",
 *  <em>9th Workshop on Artificial Intelligence and Statistics</em>,
 *  http://www.eecs.berkeley.edu/~wainwrig/Papers/WJW_AIStat03.pdf
 *
 *  \anchor YFW05 \ref YFW05
 *  J. S. Yedidia and W. T. Freeman and Y. Weiss (2005):
 *  "Constructing Free-Energy Approximations and Generalized Belief Propagation Algorithms",
 *  <em>IEEE Transactions on Information Theory</em> 51(7):2282-2312,
 *  http://ieeexplore.ieee.org/xpl/freeabs_all.jsp?arnumber=1459044
 */


/** \page discussion Ideas not worth exploring
 *  \section discuss_extendedgraphs Extended factorgraphs/regiongraphs
 *
 *  A FactorGraph and a RegionGraph are often equipped with
 *  additional properties for nodes and edges. The code to initialize those
 *  is often quite similar. Maybe one could abstract this, e.g.:
 *  \code
 *  template <typename Node1Properties, typename Node2Properties, typename EdgeProperties>
 *  class ExtFactorGraph : public FactorGraph {
 *      public:
 *          std::vector<Node1Properties>              node1Props;
 *          std::vector<Node2Properties>              node2Props;
 *          std::vector<std::vector<EdgeProperties> > edgeProps;
 *         // ...
 *  }
 *  \endcode
 *
 *  Advantages:
 *  - Less code duplication.
 *  - Easier maintainability.
 *  - Easier to write new inference algorithms.
 *
 *  Disadvantages:
 *  - Cachability may be worse.
 *  - A problem is the case where there are no properties for either type of nodes or for edges.
 *    Maybe this can be solved using specializations, or using variadac template arguments?
 *    Another possible solution would be to define a "class Empty {}", and add some code
 *    that checks for the typeid, comparing it with Empty, and doing something special in that case
 *    (e.g., not allocating memory).
 *  - The main disadvantage of this approach seems to be that it leads to even more entanglement.
 *    Therefore this is probably a bad idea.
 *
 *  \section discuss_templates Polymorphism by template parameterization
 *
 *  Instead of polymorphism by inheritance, use polymorphism by template parameterization.
 *  For example, the real reason for introducing the complicated inheritance scheme of dai::InfAlg
 *  was for functions like dai::calcMarginal. Instead, one could use a template function:
 *  \code
 *  template<typename InfAlg>
 *  Factor calcMarginal( const InfAlg &obj, const VarSet &ns, bool reInit );
 *  \endcode
 *  This would assume that the type InfAlg supports certain methods. Ideally, one would use
 *  concepts to define different classes of inference algorithms with different capabilities,
 *  for example the ability to calculate logZ, the ability to calculate marginals, the ability to
 *  calculate bounds, the ability to calculate MAP states, etc. Then, one would use traits
 *  classes in order to be able to query the capabilities of the model. For example, one would be
 *  able to query whether the inference algorithm supports calculation of logZ.  Unfortunately,
 *  this is compile-time polymorphism, whereas tests/testdai needs runtime polymorphism.
 *  Therefore this is probably a bad idea.
 */
