/*============================================================================

Title   -  MAGIC: Marching Cubes Isosurface Uncertainty Visualization for Gaussian Uncertain Data with Spatial Correlation
Authors -  Tushar M. Athawale, Kenneth Moreland, David Pugmire, Chris R. Johnson, Paul Rosen, Matthew Norman, Antigoni Georgiadou, and Alireza Entezari
Date    -  Oct 20, 2025

This code is distributed under the OSI-approved BSD 3-clause License. See LICENSE.txt for details.
 
============================================================================*/

#include <viskores/io/VTKDataSetReader.h>
#include <viskores/io/VTKDataSetWriter.h>
#include <viskores/cont/Timer.h>
#include <viskores/cont/Initialize.h>

#include <viskores/cont/DataSet.h>
#include <viskores/cont/DataSetBuilderUniform.h>

#include "MAGIC.h"

#include <iostream>

int main(int argc, char** argv)
{
  // Template to run the MAGIC filter
  // Example: ./run_contour file_path/GaussianField.vtk 27.6
  auto opts = viskores::cont::InitializeOptions::DefaultAnyDevice;
  viskores::cont::InitializeResult config = viskores::cont::Initialize(argc, argv, opts);

  if (argc != 3) {
    std::cerr << "USAGE: " << argv[0] << " [Viskores options] <filename>.vtk <isovalue>\n\n";
    std::cerr << "Viskores options are:\n" << config.Usage;
    return 1;
  }

  if (getenv("UCV_VISKORES_BACKEND")) {
    std::cerr << "WARNING: UCV_VISKORES_BACKEND is no longer supported and will be ignored.\n";
  }
  std::cout << "Using Vikores device " << config.Device.GetName() << "\n";

  // Initialize the viskores timer
  viskores::cont::Timer timer{ config.Device };

  // Read the VTK dataset provided in the first argument
  viskores::io::VTKDataSetReader reader(argv[1]);
  viskores::cont::DataSet input = reader.ReadDataSet();
    
  // Read the isovalue provided in the second argument
  viskores::Float64 isovalue = std::stod(argv[2]);

  // Run the MAGIC filter and measure time
  MAGIC contourFilter;
  contourFilter.SetMeanField("mean");
  contourFilter.SetVarianceField("variance");
  contourFilter.SetRhoXField("rhoX");
  contourFilter.SetRhoYField("rhoY");
  contourFilter.SetRhoZField("rhoZ");
  contourFilter.SetIsoValue(isovalue);
  contourFilter.SetMergeDuplicatePoints(true);
  contourFilter.SetAddInterpolationEdgeIds(true);

  // Do "burn-in" run to let device warm up.
  viskores::cont::DataSet contour = contourFilter.Execute(input);

  constexpr viskores::IdComponent NUM_TRIALS = 1;
  viskores::Float64 totalTime = 0;
  for (viskores::IdComponent trial = 0; trial < NUM_TRIALS; ++trial)
  {
    timer.Start();
    contour = contourFilter.Execute(input);
    timer.Stop();
    viskores::Float64 trialTime = timer.GetElapsedTime();
    std::cout << "Running time (seconds): " << trialTime << std::endl;
    totalTime += trialTime;
  }
  std::cout << "\nAverage running time (seconds): " << totalTime/NUM_TRIALS << "\n";

  // Write out the result for visualization in ParaView or other software
  viskores::io::VTKDataSetWriter writer("output.vtk");
  writer.SetFileTypeToBinary();
  writer.WriteDataSet(contour);

  return 0;
}
