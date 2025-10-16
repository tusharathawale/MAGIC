/*============================================================================

Title   -  MAGIC: Marching Cubes Isosurface Uncertainty Visualization for Gaussian Uncertain Data with Spatial Correlation
Authors -  Tushar M. Athawale, Kenneth Moreland, David Pugmire, Chris R. Johnson, Paul Rosen, Matthew Norman, Antigoni Georgiadou, and Alireza Entezari
Date    -  Oct 20, 2025

This code is distributed under the OSI-approved BSD 3-clause License. See LICENSE.txt for details.
 
============================================================================*/

#include <viskores/filter/contour/Contour.h>
#include "gaussian_distribution.h"

class MAGIC : public viskores::filter::contour::AbstractContour
{
public:
  // Set the name of the mean field
  VISKORES_CONT void SetMeanField(const std::string& name)
  {
    this->SetActiveField(0, name, viskores::cont::Field::Association::Points);
  }
  // Set the name of the pointwise variance field
  VISKORES_CONT void SetVarianceField(const std::string& name)
  {
    this->SetActiveField(1, name, viskores::cont::Field::Association::Points);
  }
  // Set the name of the field for edgewise covariance in the X direction
  VISKORES_CONT void SetRhoXField(const std::string& name)
  {
    this->SetActiveField(2, name, viskores::cont::Field::Association::Points);
  }
  // Set the name of the field for edgewise covariance in the Y direction
  VISKORES_CONT void SetRhoYField(const std::string& name)
  {
    this->SetActiveField(3, name, viskores::cont::Field::Association::Points);
  }
  // Set the name of the field for edgewise covariance in the Z direction
  VISKORES_CONT void SetRhoZField(const std::string& name)
  {
    this->SetActiveField(4, name, viskores::cont::Field::Association::Points);
  }

protected:
  // The MAGIC filter
  VISKORES_CONT viskores::cont::DataSet DoExecute(const viskores::cont::DataSet &input) override;
};
