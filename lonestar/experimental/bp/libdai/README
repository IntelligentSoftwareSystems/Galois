libDAI  -  A free/open source C++ library for Discrete Approximate Inference

-------------------------------------------------------------------------------

Version:  0.3.0
Date:     July 12, 2011
See also: http://www.libdai.org

-------------------------------------------------------------------------------

License

libDAI is free software; you can redistribute it and/or modify it under the
terms of the BSD 2-clause license (also known as the FreeBSD license), which
can be found in the accompanying LICENSE file.

[Note: up to and including version 0.2.7, libDAI was licensed under the GNU
General Public License (GPL) version 2 or higher.]


-------------------------------------------------------------------------------

Citing libDAI

If you write a scientific paper describing research that made substantive use
of this library, please cite the following paper describing libDAI:

Joris M. Mooij;
libDAI: A free & open source C++ library for Discrete Approximate Inference in graphical models;
Journal of Machine Learning Research, 11(Aug):2169-2173, 2010.

In BiBTeX format (for your convenience):

  @article{Mooij_libDAI_10,
    author    = {Joris M. Mooij},
    title     = {lib{DAI}: A Free and Open Source {C++} Library for Discrete Approximate Inference in Graphical Models},
    journal   = {Journal of Machine Learning Research},
    year      = 2010,
    month     = Aug,
    volume    = 11,
    pages     = {2169-2173},
    url       = "http://www.jmlr.org/papers/volume11/mooij10a/mooij10a.pdf"
  }

Moreover, as a personal note, I would appreciate it to be informed about any
publications using libDAI at joris dot mooij at libdai dot org.

-------------------------------------------------------------------------------

About libDAI

libDAI is a free/open source C++ library that provides implementations of
various (approximate) inference methods for discrete graphical models. libDAI
supports arbitrary factor graphs with discrete variables; this includes
discrete Markov Random Fields and Bayesian Networks.

The library is targeted at researchers. To be able to use the library, a good
understanding of graphical models is needed.

The best way to use libDAI is by writing C++ code that invokes the library; in
addition, part of the functionality is accessibly by using the

  * command line interface
  * (limited) MatLab interface
  * (experimental) python interface
  * (experimental) octave interface.

libDAI can be used to implement novel (approximate) inference algorithms and to
easily compare the accuracy and performance with existing algorithms that have
been implemented already.

A solver using libDAI was amongst the three winners of the UAI 2010 Approximate
Inference Challenge (see http://www.cs.huji.ac.il/project/UAI10/ for more
information). The full source code is provided as part of the library.

Features

Currently, libDAI supports the following (approximate) inference methods:

  * Exact inference by brute force enumeration;
  * Exact inference by junction-tree methods;
  * Mean Field;
  * Loopy Belief Propagation [KFL01];
  * Fractional Belief Propagation [WiH03];
  * Tree-Reweighted Belief Propagation [WJW03];
  * Tree Expectation Propagation [MiQ04];
  * Generalized Belief Propagation [YFW05];
  * Double-loop GBP [HAK03];
  * Various variants of Loop Corrected Belief Propagation [MoK07, MoR05];
  * Gibbs sampler;
  * Conditioned Belief Propagation [EaG09];
  * Decimation algorithm.

These inference methods can be used to calculate partition sums, marginals over
subsets of variables, and MAP states (the joint state of variables that has
maximum probability).

In addition, libDAI supports parameter learning of conditional probability
tables by Expectation Maximization.

Limitations

libDAI is not intended to be a complete package for approximate inference.
Instead, it should be considered as an "inference engine", providing various
inference methods. In particular, it contains no GUI, currently only supports
its own file format for input and output (although support for standard file
formats may be added later), and provides very limited visualization
functionalities. The only learning method supported currently is Expectation
Maximization (or Maximum Likelihood if no data is missing) for learning factor
parameters.

Rationale

In my opinion, the lack of open source "reference" implementations hampers
progress in research on approximate inference. Methods differ widely in terms
of quality and performance characteristics, which also depend in different ways
on various properties of the graphical models. Finding the best approximate
inference method for a particular application therefore often requires
empirical comparisons. However, implementing and debugging these methods takes
a lot of time which could otherwise be spent on research. I hope that this code
will aid researchers to be able to easily compare various (existing as well as
new) approximate inference methods, in this way accelerating research and
stimulating real-world applications of approximate inference.

Language

Because libDAI is implemented in C++, it is very fast compared with
implementations in MatLab (a factor 1000 faster is not uncommon). libDAI does
provide a (limited) MatLab interface for easy integration with MatLab. It also
provides a command line interface and experimental python and octave interfaces
(thanks to Patrick Pletscher).

Compatibility

The code has been developed under Debian GNU/Linux with the GCC compiler suite.
libDAI compiles successfully with g++ versions 3.4 up to 4.6.

libDAI has also been successfully compiled with MS Visual Studio 2008 under
Windows (but not all build targets are supported yet) and with Cygwin under
Windows.

Finally, libDAI has been compiled successfully on MacOS X.

Downloading libDAI

The libDAI sources and documentation can be downloaded from the libDAI website:
http://www.libdai.org.

Mailing list

The Google group "libDAI" (http://groups.google.com/group/libdai) can be used
for getting support and discussing development issues.

-------------------------------------------------------------------------------

Building libDAI under UNIX variants (Linux / Cygwin / Mac OS X)

Preparations

You need:

  * a recent version of gcc (at least version 3.4)
  * GNU make
  * recent boost C++ libraries (at least version 1.37; however, version 1.37
    shipped with Ubuntu 9.04 is known not to work)
  * GMP library (or the Windows port called MPIR)
  * doxygen (only for building the documentation)
  * graphviz (only for using some of the libDAI command line utilities)
  * CImg library (only for building the image segmentation example)

On Debian/Ubuntu, you can easily install the required packages with a single
command:

  apt-get install g++ make doxygen graphviz libboost-dev libboost-graph-dev libboost-program-options-dev libboost-test-dev libgmp-dev cimg-dev

(root permissions needed).

On Mac OS X (10.4 is known to work), these packages can be installed easily via
MacPorts. If MacPorts is not already installed, install it according to the
instructions at http://www.macports.org/. Then, a simple

  sudo port install gmake boost gmp doxygen graphviz

should be enough to install everything that is needed.

On Cygwin, the prebuilt Cygwin package boost-1.33.1-x is known not to work. You
can however obtain the latest boost version (you need at least 1.37.0) from
http://www.boost.org/ and build it as described in the next subsection.

Building boost under Cygwin

  * Download the latest boost libraries from http://www.boost.org
  * Build the required boost libraries using:

        ./bootstrap.sh --with-libraries=program_options,math,graph,test --prefix=/boost_root/
        ./bjam

  * In order to use dynamic linking, the boost .dll's should be somewhere in
    the path. This can be achieved by a command like:

        export PATH=$PATH:/boost_root/stage/lib

Building libDAI

To build the libDAI source, first copy a template Makefile.* to Makefile.conf
(for example, copy Makefile.LINUX to Makefile.conf if you use GNU/Linux). Then,
edit the Makefile.conf template to adapt it to your local setup. In case you
want to use Boost libraries which are installed in non-standard locations, you
have to tell the compiler and linker about their locations (using the -I, -L
flags for GCC; also you may need to set the LD_LIBRARY_PATH environment
variable correctly before running libDAI binaries). Platform independent build
options can be set in Makefile.ALL. Finally, run

  make

The build includes a regression test, which may take a while to complete.

If the build is successful, you can test the example program:

  examples/example tests/alarm.fg

or the more extensive test program:

  tests/testdai --aliases tests/aliases.conf --filename tests/alarm.fg --methods JTREE_HUGIN BP_SEQMAX

-------------------------------------------------------------------------------

Building libDAI under Windows

Preparations

You need:

  * A recent version of MicroSoft Visual Studio (2008 is known to work)
  * recent boost C++ libraries (version 1.37 or higher)
  * GMP or MPIR library
  * GNU make (can be obtained from http://gnuwin32.sourceforge.net)
  * CImg library (only for building the image segmentation example)

For the regression test, you need:

  * GNU diff, GNU sed (can be obtained from http://gnuwin32.sourceforge.net)

Building boost under Windows

Because building boost under Windows is tricky, I provide some guidance here.

  * Download the boost zip file from http://www.boost.org/users/download and
    unpack it somewhere.
  * Download the bjam executable from http://www.boost.org/users/download and
    unpack it somewhere else.
  * Download Boost.Build (v2) from http://www.boost.org/docs/tools/build/
    index.html and unpack it yet somewhere else.
  * Edit the file boost-build.jam in the main boost directory to change the
    BOOST_BUILD directory to the place where you put Boost.Build (use UNIX /
    instead of Windows \ in pathnames).
  * Copy the bjam.exe executable into the main boost directory. Now if you
    issue "bjam --version" you should get a version and no errors. Issueing
    "bjam --show-libraries" will show the libraries that will be built.
  * The following command builds the boost libraries that are relevant for
    libDAI:

        bjam --with-graph --with-math --with-program_options --with-test link=static runtime-link=shared

Building GMP or MPIR under Windows

Information about how to build GPR or MPIR under Windows can be found on the
internet. The user has to update Makefile.WINDOWS in order to link with the GPR
/MPIR libraries.

Building libDAI

To build the source, copy Makefile.WINDOWS to Makefile.conf. Then, edit
Makefile.conf to adapt it to your local setup. Platform independent build
options can be set in Makefile.ALL. Finally, run (from the command line)

  make

The build includes a regression test, which may take a while to complete.

If the build is successful, you can test the example program:

  examples\example tests\alarm.fg

or the more extensive test program:

  tests\testdai --aliases tests\aliases.conf --filename tests\alarm.fg --methods JTREE_HUGIN BP_SEQMAX

-------------------------------------------------------------------------------

Building the libDAI MatLab interface

You need:

  * MatLab
  * The platform-dependent requirements described above

First, you need to build the libDAI source as described above for your
platform. By default, the MatLab interface is disabled, so before compiling the
source, you have to enable it in Makefile.ALL by setting

  WITH_MATLAB=true

Also, you have to configure the MatLab-specific parts of Makefile.conf to match
your system (in particular, the Makefile variables ME, MATLABDIR and MEX). The
MEX file extension depends on your platform; for a 64-bit linux x86_64 system
this would be "ME=.mexa64", for a 32-bit linux x86 system "ME=.mexglx". If you
are unsure about your MEX file extension: it needs to be the same as what the
MatLab command "mexext" returns. The required MEX files are built by issuing

  make

from the command line. The MatLab interface is much less powerful than using
libDAI from C++. There are two reasons for this: (i) it is boring to write MEX
files; (ii) the large performance penalty paid when large data structures (like
factor graphs) have to be converted between their native C++ data structure to
something that MatLab understands.

A simple example of how to use the MatLab interface is the following (entered
at the MatLab prompt), which performs exact inference by the junction tree
algorithm and approximate inference by belief propagation on the ALARM network:

  cd path_to_libdai/matlab
  [psi] = dai_readfg ('../tests/alarm.fg');
  [logZ,q,md,qv,qf] = dai (psi, 'JTREE', '[updates=HUGIN,verbose=0]')
  [logZ,q,md,qv,qf] = dai (psi, 'BP', '[updates=SEQMAX,tol=1e-9,maxiter=10000,logdomain=0]')

where "path_to_libdai" has to be replaced with the directory in which libDAI
was installed. For other algorithms and some default parameters, see the file
tests/aliases.conf.

-------------------------------------------------------------------------------

Building the documentation

Install doxygen, graphviz and a TeX distribution and use

  make doc

to build the documentation. If the documentation is not clear enough, feel free
to send me an email (or even better, to improve the documentation and send a
patch!). The documentation can also be browsed online at http://www.libdai.org.

