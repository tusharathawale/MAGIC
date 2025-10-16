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

#include "MAGIC.h"

#include <iostream>

std::string backend = "serial";

void initBackend(viskores::cont::Timer &timer)
{
    // init the vtkh device
    char const *tmp = getenv("UCV_VISKORES_BACKEND");

    if (tmp == nullptr)
    {
        return;
    }
    else
    {
        backend = std::string(tmp);
        std::cout << "Setting the device with UCV_VISKORES_BACKEND=" << backend << "\n";
        std::cout << "This method is antiquated. Consider using the --vtkm-device command line argument." << std::endl;
    }

    std::cout << "vtkm backend is:" << backend << std::endl;

    if (backend == "serial")
    {
        viskores::cont::RuntimeDeviceTracker &device_tracker = viskores::cont::GetRuntimeDeviceTracker();
        device_tracker.ForceDevice(viskores::cont::DeviceAdapterTagSerial());
        timer.Reset(viskores::cont::DeviceAdapterTagSerial());
    }
    else if (backend == "openmp")
    {
        viskores::cont::RuntimeDeviceTracker &device_tracker = viskores::cont::GetRuntimeDeviceTracker();
        device_tracker.ForceDevice(viskores::cont::DeviceAdapterTagOpenMP());
        timer.Reset(viskores::cont::DeviceAdapterTagOpenMP());
    }
    else if (backend == "cuda")
    {
        viskores::cont::RuntimeDeviceTracker &device_tracker = viskores::cont::GetRuntimeDeviceTracker();
        device_tracker.ForceDevice(viskores::cont::DeviceAdapterTagCuda());
        timer.Reset(viskores::cont::DeviceAdapterTagCuda());
    }
    else
    {
        std::cerr << " unrecognized backend " << backend << std::endl;
    }
    return;
}


int main(int argc, char** argv)
{
    
  // Template to run MAGIC filter
  // Example: ./run_contour file_path/GaussianField.vtk 27.6
  if (argc != 3) {
    std::cerr << "USAGE: " << argv[0] << " <filename>.vtk <isovalue>\n";
    return 1;
  }
    
  // Initialize the viskores timer with a proper backend
  auto opts = viskores::cont::InitializeOptions::DefaultAnyDevice;
  viskores::cont::InitializeResult config = viskores::cont::Initialize(argc, argv, opts);
  viskores::cont::Timer timer{ config.Device };
  initBackend(timer);

  // Read the VTK dataset provided in the first argument
  viskores::io::VTKDataSetReader reader(argv[1]);
  viskores::cont::DataSet input = reader.ReadDataSet();
    
  // Read the isovalue provided in the second argument
  viskores::Float64 isovalue = std::stod(argv[2]);

  // Run the MAGIC filter and measure time
  MAGIC contourFilter;
  contourFilter.SetMeanField("mean");
  contourFilter.SetVarianceField("variance");
  contourFilter.SetIsoValue(isovalue);
  contourFilter.SetMergeDuplicatePoints(true);
  contourFilter.SetAddInterpolationEdgeIds(true);
    
  timer.Start();
  viskores::cont::DataSet contour = contourFilter.Execute(input);
  timer.Stop();
  std::cout << "Running time: " << timer.GetElapsedTime() << std::endl;
  
  // Write out the result for visualization in ParaView or other software
  viskores::io::VTKDataSetWriter writer("output.vtk");
  writer.SetFileTypeToBinary();
  writer.WriteDataSet(contour);

  return 0;
}
