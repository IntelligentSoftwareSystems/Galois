/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 */

/**
 * Femap.h: Classes for parsing and writing Femap Neutral file format
 * DG++
 *
 * Copyright (c) 2006 Adrian Lew
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject
 * to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
#ifndef _FEMAP_H
#define _FEMAP_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>

#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <cmath>

#ifdef GALOIS_HAS_GZ_SUPP
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/copy.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#endif

#include "FemapData.h"

class Femap {
public:
  size_t getNumMaterials() const { return _materials.size(); }
  size_t getNumProperties() const { return _properties.size(); }
  size_t getNumNodes() const { return _nodes.size(); }
  size_t getNumElements() const { return _elements.size(); }
  //! no. of elements with topology indicator t
  //! @param t
  size_t getNumElements(size_t t) const;
  size_t getNumConstraintSets() const { return _constraintSets.size(); }
  size_t getNumLoadSets() const { return _loadSets.size(); }
  size_t getNumGroups() const { return _groups.size(); }

  const std::vector<femapMaterial>& getMaterials() const { return _materials; }
  const std::vector<femapProperty>& getProperties() const {
    return _properties;
  }
  const std::vector<femapNode>& getNodes() const { return _nodes; }
  const std::vector<femapElement>& getElements() const { return _elements; }
  const std::vector<femapConstraintSet>& getConstraintSets() const {
    return _constraintSets;
  }
  const std::vector<femapLoadSet>& getLoadSets() const { return _loadSets; }
  const std::vector<femapGroup>& getGroups() const { return _groups; }

  //! get elements with topology indicator t
  //! @param t
  //! @param vout: output vector
  void getElements(size_t t, std::vector<femapElement>& vout)
      const; // gets elements with topology t

  const femapMaterial& getMaterial(size_t i) const { return _materials[i]; }
  const femapProperty& getProperty(size_t i) const { return _properties[i]; }
  const femapNode& getNode(size_t i) const { return _nodes[i]; }
  const femapElement& getElement(size_t i) const { return _elements[i]; }
  const femapConstraintSet& getConstraintSet(size_t i) const {
    return _constraintSets[i];
  }
  const femapLoadSet& getLoadSet(size_t i) const { return _loadSets[i]; }
  const femapGroup& getGroup(size_t i) const { return _groups[i]; }

  int getMaterialId(size_t i) const { return _materialIdMap.at(i); }
  int getPropertyId(size_t i) const { return _propertyIdMap.at(i); }
  int getNodeId(size_t i) const { return _nodeIdMap.at(i); }
  int getElementId(size_t i) const { return _elementIdMap.at(i); }
  int getConstraintSetId(size_t i) const { return _constraintSetIdMap.at(i); }
  int getLoadSetId(size_t i) const { return _loadSetIdMap.at(i); }
  int getGroupId(size_t i) const { return _groupIdMap.at(i); }

protected:
  std::vector<femapMaterial> _materials;
  std::vector<femapProperty> _properties;
  std::vector<femapNode> _nodes;
  std::vector<femapElement> _elements;
  std::vector<femapGroup> _groups;
  std::vector<femapConstraintSet> _constraintSets;
  std::vector<femapLoadSet> _loadSets;

  std::map<size_t, size_t> _materialIdMap;
  std::map<size_t, size_t> _propertyIdMap;
  std::map<size_t, size_t> _nodeIdMap;
  std::map<size_t, size_t> _elementIdMap;
  std::map<size_t, size_t> _groupIdMap;
  std::map<size_t, size_t> _constraintSetIdMap;
  std::map<size_t, size_t> _loadSetIdMap;

private:
};

class FemapInput : public Femap {
public:
  FemapInput(const char* fileName);

private:
#ifdef GALOIS_HAS_GZ_SUPP
  boost::iostreams::filtering_istream m_ifs;
#else
  std::ifstream m_ifs;
#endif

  void _readHeader();
  void _readProperty();
  void _readNodes();
  void _readElements();
  void _readGroups();
  void _readConstraints();
  void _readLoads();
  void _readMaterial();
  void nextLine(int);
  void nextLine();
};

class FemapOutput : public Femap {
public:
  FemapOutput(const char* fileName) : _ofs(fileName) {}

private:
  std::ofstream _ofs;
};
#endif //_FEMAP_H
