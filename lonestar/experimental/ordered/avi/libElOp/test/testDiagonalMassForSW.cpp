/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism.
 * The code is being released under the terms of the 3-Clause
 * BSD License (a
 * copy is located in LICENSE.txt at the top-level
 * directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All
 * rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES
 * CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF
 * MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND
 * WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE
 * FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS
 * OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under
 * no circumstances
 * shall University be liable for incidental, special,
 * indirect, direct or
 * consequential damages or loss of profits, interruption
 * of business, or
 * related expenses which may arise from use of Software or
 * Documentation,
 * including but not limited to those resulting from defects
 * in Software and/or
 * Documentation, or loss or inaccuracy of data of any
 * kind.
 */

// Sriramajayam

#include "P12DElement.h"
#include "DiagonalMassForSW.h"

int main() {
  // Create a P12D element.
  double coord[] = {0., 0., sqrt(2), sqrt(3), sqrt(3), sqrt(2)};
  std::vector<double> coordinates;
  coordinates.assign(coord, coord + 6);

  Segment<2>::SetGlobalCoordinatesArray(coordinates);
  Triangle<2>::SetGlobalCoordinatesArray(coordinates);

  Element* Elm = new P12D<2>::Bulk(1, 2, 3);

  // Create simple material.
  IsotropicLinearElastic ILE(1.0, 1.0, 1.0); // lambda, mu and ref_rho.

  // Create Residue object to compute diagonal mass vector.
  DiagonalMassForSW MassVec(Elm, ILE, 0, 1);

  // Test class DiagonalMassForSW
  std::cout << "\nNumber of fields: " << MassVec.getFields().size()
            << " should be 2.";
  std::cout << "\nLocal fields: " << MassVec.getFields()[0] << ","
            << MassVec.getFields()[1] << " should be 0,1.";
  std::cout << "\nNumber of dofs per field: " << MassVec.getFieldDof(0) << ","
            << MassVec.getFieldDof(1) << " should be 3,3.";
  std::cout << "\nMaterial given over element: "
            << MassVec.GetSimpleMaterial().getMaterialName()
            << " should be IsotropicLinearElastic.";
  std::cout << "\nDensity of material in reference config.: "
            << MassVec.GetSimpleMaterial().getDensityInReference()
            << " should be 1.";

  // Testing calculation of mass vector:
  std::vector<std::vector<double>> argval(2);
  argval[0].resize(3, 0.);
  argval[1].resize(3, 0.);

  std::vector<std::vector<double>> funcval;
  if (!MassVec.getVal(argval, &funcval)) {
    std::cerr << "\nCould not compute mass vector. Test failed.\n";
    exit(1);
  }
  for (unsigned int f = 0; f < MassVec.getFields().size(); f++) {
    std::cout << "\nMass vector for field " << f << ": (";
    for (int a = 0; a < MassVec.getFieldDof(f); a++)
      std::cout << funcval[f][a] << ",";
    std::cout << ")\n";
  }
  // Exact vector is computed as the volume of a tetrahedron- each entry is
  // (1/3)*rho_0*area of triangle*ht(=1).
  double Area = 0.;
  for (unsigned int q = 0; q < Elm->getIntegrationWeights(0).size(); q++)
    Area += Elm->getIntegrationWeights(0)[q];

  double mv = (1. / 3.) * MassVec.GetSimpleMaterial().getDensityInReference() *
              Area * 1.0;
  std::cout << "\nBoth mass vectors should read: (" << mv << "," << mv << ","
            << mv << ")\n";

  std::cout << "\nTesting Copy constructor: ";
  {
    DiagonalMassForSW MassVecCopy(MassVec);
    std::cout << "\nNumber of fields: " << MassVecCopy.getFields().size()
              << " should be 2.";
    std::cout << "\nLocal fields: " << MassVecCopy.getFields()[0] << ","
              << MassVecCopy.getFields()[1] << " should be 0,1.";
    std::cout << "\nNumber of dofs per field: " << MassVecCopy.getFieldDof(0)
              << "," << MassVecCopy.getFieldDof(1) << " should be 3,3.";
    std::cout << "\nMaterial given over element: "
              << MassVecCopy.GetSimpleMaterial().getMaterialName()
              << " should be IsotropicLinearElastic.";
    std::cout << "\nDensity of material in reference config.: "
              << MassVecCopy.GetSimpleMaterial().getDensityInReference()
              << " should be 1.";
  }

  std::cout << "\n\nTesting Cloning: ";
  {
    DiagonalMassForSW* MassVecClone = MassVec.clone();
    std::cout << "\nNumber of fields: " << MassVecClone->getFields().size()
              << " should be 2.";
    std::cout << "\nLocal fields: " << MassVecClone->getFields()[0] << ","
              << MassVecClone->getFields()[1] << " should be 0,1.";
    std::cout << "\nNumber of dofs per field: " << MassVecClone->getFieldDof(0)
              << "," << MassVecClone->getFieldDof(1) << " should be 3,3.";
    std::cout << "\nMaterial given over element: "
              << MassVecClone->GetSimpleMaterial().getMaterialName()
              << " should be IsotropicLinearElastic.";
    std::cout << "\nDensity of material in reference config.: "
              << MassVecClone->GetSimpleMaterial().getDensityInReference()
              << " should be 1.";
    delete MassVecClone;
  }

  std::cout << "\n\nTesing finished.\n";
}
