/*  This file is part of libDAI - http://www.libdai.org/
 *
 *  Copyright (c) 2006-2011, The libDAI authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
 */


/** \file
 *  \brief Main libDAI header file. It \#includes all other libDAI headers.
 * 
 *  \todo Replace VarSets by SmallSet<size_t> where appropriate, in order to minimize the use of FactorGraph::findVar().
 *
 *  \todo Improve SWIG interfaces and merge their build process with the main build process
 */


#ifndef __defined_libdai_alldai_h
#define __defined_libdai_alldai_h


#include <string>
#include <dai/daialg.h>
#include <dai/properties.h>
#include <dai/exactinf.h>
#include <dai/evidence.h>
#include <dai/emalg.h>
#ifdef DAI_WITH_BP
    #include <dai/bp.h>
#endif
#ifdef DAI_WITH_FBP
    #include <dai/fbp.h>
#endif
#ifdef DAI_WITH_TRWBP
    #include <dai/trwbp.h>
#endif
#ifdef DAI_WITH_MF
    #include <dai/mf.h>
#endif
#ifdef DAI_WITH_HAK
    #include <dai/hak.h>
#endif
#ifdef DAI_WITH_LC
    #include <dai/lc.h>
#endif
#ifdef DAI_WITH_TREEEP
    #include <dai/treeep.h>
#endif
#ifdef DAI_WITH_JTREE
    #include <dai/jtree.h>
#endif
#ifdef DAI_WITH_MR
    #include <dai/mr.h>
#endif
#ifdef DAI_WITH_GIBBS
    #include <dai/gibbs.h>
#endif
#ifdef DAI_WITH_CBP
    #include <dai/cbp.h>
#endif
#ifdef DAI_WITH_DECMAP
    #include <dai/decmap.h>
#endif


/// Namespace for libDAI
namespace dai {


/// Returns a map that contains for each built-in inference algorithm its name and a pointer to an object of that type
/** \obsolete This functionality is obsolete and will be removed in future versions of libDAI
 */
std::map<std::string, InfAlg *>& builtinInfAlgs();


/// Returns a set of names of all available inference algorithms
/*  These are the names of the algorithms that were compiled in and can be 
 *  given to \ref newInfAlg and \ref newInfAlgFromString.  
 *  \return A set of strings, each one corresponding with the name of an available inference algorithm.
 *  \note The set is returned by value because it will be reasonably small 
 *  enough and this function is expected to be called infrequently.
 */
std::set<std::string> builtinInfAlgNames();


/// Constructs a new inference algorithm.
/** \param name The name of the inference algorithm.
 *  \param fg The FactorGraph that the algorithm should be applied to.
 *  \param opts A PropertySet specifying the options for the algorithm.
 *  \return Returns a pointer to the new InfAlg object; it is the responsibility of the caller to delete it later.
 *  \throw UNKNOWN_DAI_ALGORITHM if the requested name is not known/compiled in.
 */
InfAlg *newInfAlg( const std::string &name, const FactorGraph &fg, const PropertySet &opts );


/// Constructs a new inference algorithm.
/** \param nameOpts The name and options of the inference algorithm (should be in the format "name[key1=val1,key2=val2,...,keyn=valn]").
 *  \param fg The FactorGraph that the algorithm should be applied to.
 *  \return Returns a pointer to the new InfAlg object; it is the responsibility of the caller to delete it later.
 *  \throw UNKNOWN_DAI_ALGORITHM if the requested name is not known/compiled in.
 */
InfAlg *newInfAlgFromString( const std::string &nameOpts, const FactorGraph &fg );


/// Constructs a new inference algorithm.
/** \param nameOpts The name and options of the inference algorithm (should be in the format "name[key1=val1,key2=val2,...,keyn=valn]").
 *  \param fg The FactorGraph that the algorithm should be applied to.
 *  \param aliases Maps names to strings in the format "name[key1=val1,key2=val2,...,keyn=valn]"; if not empty, alias substitution
 *  will be performed when parsing \a nameOpts by invoking parseNameProperties(const std::string &,const std::map<std::string,std::string> &)
 *  \see newInfAlgFromString(const std::string &, const FactorGraph &)
 */
InfAlg *newInfAlgFromString( const std::string &nameOpts, const FactorGraph &fg, const std::map<std::string,std::string> &aliases );


/// Extracts the name and property set from a string \a s in the format "name[key1=val1,key2=val2,...]" or "name"
std::pair<std::string, PropertySet> parseNameProperties( const std::string &s );


/// Extracts the name and property set from a string \a s in the format "name[key1=val1,key2=val2,...]" or "name", performing alias substitution
/** Alias substitution is performed as follows: as long as name appears as a key in \a aliases,
 *  it is substituted by its value. Properties in \a s override those of the alias (in case of
 *  recursion, the "outer" properties override those of the "inner" aliases).
 */
std::pair<std::string, PropertySet> parseNameProperties( const std::string &s, const std::map<std::string,std::string> &aliases );


/// Reads aliases from file named \a filename
/** \param filename Name of the alias file
 *  \return A map that maps aliases to the strings they should be substituted with.
 *  \see \ref fileformats-aliases
 */
std::map<std::string,std::string> readAliasesFile( const std::string &filename );


} // end of namespace dai


/** \example example.cpp
 *  This example illustrates how to read a factor graph from a file and how to
 *  run several inference algorithms (junction tree, loopy belief propagation,
 *  and the max-product algorithm) on it.
 */


/** \example example_imagesegmentation.cpp
 *  This example shows how one can use approximate inference in factor graphs
 *  on a simple vision task: given two images, identify smooth regions where these
 *  two images differ more than some threshold. This can be used to seperate 
 *  foreground from background if one image contains the background and the other
 *  one the combination of background and foreground.
 *
 *  \note In order to build this example, a recent version of CImg needs to be installed.
 */


/** \example uai2010-aie-solver.cpp
 *  This example contains the full source code of the solver that was one of the
 *  winners (the 'libDAI2' solver) in the UAI 2010 Approximate Inference Challenge
 *  (see http://www.cs.huji.ac.il/project/UAI10/ for more information).
 */


#endif
