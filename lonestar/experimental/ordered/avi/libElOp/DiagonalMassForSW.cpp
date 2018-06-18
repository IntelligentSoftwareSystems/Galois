// Sriramajayam

#include "DiagonalMassForSW.h"
#include "Material.h"

bool DiagonalMassForSW::getVal(const MatDouble& argval,
                               MatDouble& funcval) const {
  size_t Dim = fieldsUsed.size();

  // Assume that all fields use the same quadrature rules
  size_t nquad = element.getIntegrationWeights(fieldsUsed[0]).size();

  VecSize_t nDof(Dim, 0);    // Number of dofs in each field
  MatDouble IntWeights(Dim); // Integration weights for each field
  MatDouble Shape(Dim);      // Shape functions for each field.

  for (size_t f = 0; f < Dim; f++) {
    nDof[f]       = element.getDof(fieldsUsed[f]);
    IntWeights[f] = element.getIntegrationWeights(fieldsUsed[f]);
    Shape[f]      = element.getShapes(fieldsUsed[f]);
  }

  // Resize funcval if required.
  if (funcval.size() < fieldsUsed.size()) {
    funcval.resize(fieldsUsed.size());
  }

  for (size_t f = 0; f < fieldsUsed.size(); f++) {
    if (funcval[f].size() < nDof[f]) {
      funcval[f].resize(nDof[f], 0.);
    } else {
      for (size_t a = 0; a < nDof[f]; a++) {
        funcval[f][a] = 0.;
      }
    }
  }

  for (size_t q = 0; q < nquad; q++) {
    VecDouble F(SimpleMaterial::MAT_SIZE, 0.); // F = I in the reference config.

    std::copy(SimpleMaterial::I_MAT,
              SimpleMaterial::I_MAT + SimpleMaterial::MAT_SIZE, F.begin());

    //    F[0] = 1.;
    //    F[4] = 1.;
    //    F[8] = 1.;

    double Ref_rho = 0.;
    if (!material.getLocalMaterialDensity(&F, Ref_rho)) {
      std::cerr << "\nDiagonalMassForSW::GetVal()- Could not compute local "
                   "density.\n";
      return false;
    }

    for (size_t f = 0; f < fieldsUsed.size(); f++) {
      for (size_t a = 0; a < nDof[f]; a++) {
        funcval[f][a] += IntWeights[f][q] * Ref_rho * Shape[f][nDof[f] * q + a];
      }
    }
  }

  return true;
}
