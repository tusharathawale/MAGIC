/*============================================================================

Title   -  MAGIC: Marching Cubes Isosurface Uncertainty Visualization for Gaussian Uncertain Data with Spatial Correlation
Authors -  Tushar M. Athawale, Kenneth Moreland, David Pugmire, Chris R. Johnson, Paul Rosen, Matthew Norman, Antigoni Georgiadou, and Alireza Entezari
Date    -  Oct 20, 2025
 
This code implements the following previous paper using the Viskores library:

T. Athawale and A. Entezari, "Uncertainty Quantification in Linear Interpolation for Isosurface Extraction,"
in IEEE Transactions on Visualization and Computer Graphics, vol. 19, no. 12, pp. 2723-2732, Dec. 2013,
doi: 10.1109/TVCG.2013.208.
 
This code is distributed under the OSI-approved BSD 3-clause License. See LICENSE.txt for details.
 
============================================================================*/

#include <viskores/filter/contour/Contour.h>
#include "uniform_even_z_density.h"

class MCA_INDEPENDENT_UNIFORM : public viskores::filter::contour::AbstractContour
{
public:
  // Set the name of the mean field
  VISKORES_CONT void SetMeanField(const std::string& name)
  {
    this->SetActiveField(0, name, viskores::cont::Field::Association::Points);
  }
  
  // Set the name of the variance (width of the uniform distribution) field
  VISKORES_CONT void SetVarianceField(const std::string& name)
  {
    this->SetActiveField(1, name, viskores::cont::Field::Association::Points);
  }

protected:
  // The INDEPENDENT UNIFORM noise filter
  VISKORES_CONT viskores::cont::DataSet DoExecute(const viskores::cont::DataSet &input) override;
};
