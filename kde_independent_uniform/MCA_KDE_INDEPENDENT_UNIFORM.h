/*============================================================================

Title   -  MAGIC: Marching Cubes Isosurface Uncertainty Visualization for Gaussian Uncertain Data with Spatial Correlation
Authors -  Tushar M. Athawale, Kenneth Moreland, David Pugmire, Chris R. Johnson, Paul Rosen, Matthew Norman, Antigoni Georgiadou, and Alireza Entezari
Date    -  Oct 20, 2025
 
This code implements the following previous paper using the Viskores library:

 T. Athawale, E. Sakhaee and A. Entezari, "Isosurface Visualization of Data with Nonparametric Models for Uncertainty,"
 in IEEE Transactions on Visualization and Computer Graphics, vol. 22, no. 1, pp. 777-786, 31 Jan. 2016,
 doi: 10.1109/TVCG.2015.2467958.

This code is distributed under the OSI-approved BSD 3-clause License. See LICENSE.txt for details.
 
============================================================================*/

#include <viskores/filter/contour/Contour.h>
#include "kdeWithUniformKernel.h"

class MCA_KDE_INDEPENDENT_UNIFORM : public viskores::filter::contour::AbstractContour
{
public:
    
  VISKORES_CONT void SetEnsembleField(const std::string& name)
  {
    this->SetActiveField(0, name, viskores::cont::Field::Association::Points);
  }
  
protected:
  VISKORES_CONT viskores::cont::DataSet DoExecute(const viskores::cont::DataSet &input) override;
};
