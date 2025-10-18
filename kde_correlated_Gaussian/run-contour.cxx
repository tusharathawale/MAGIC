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

#include <viskores/cont/DataSetBuilderUniform.h>
#include <viskores/cont/ArrayHandleRuntimeVec.h>
#include <viskores/cont/ArrayCopy.h>

#include "KDE_MAGIC.h"

#include <iostream>

std::string backend = "serial";
using SupportedTypesVec = viskores::List<viskores::Vec<float, 7>>;

void initBackend(viskores::cont::Timer &timer)
{
    // Initialize the viskores device
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
  
  // Template to run the KDE MAGIC filter
  // Example: ./run-contour file_path/tangle_correlation_one_ tangle 64 64 64 14 27.6
  if (argc != 8)
  {
    std::cout << "<executable> <SyntheticDataSuffix> <FieldName> <Dimx> <Dimy> <Dimz> <num of ensembles> <isovalue>" << std::endl;
    exit(0);
  }
    
  // Initialize the viskores timer with a proper backend
  auto opts = viskores::cont::InitializeOptions::DefaultAnyDevice;
  viskores::cont::InitializeResult config = viskores::cont::Initialize(argc, argv, opts);
  viskores::cont::Timer timer{ config.Device };
  initBackend(timer);

  // Read all ensemble members
  std::string dataPathSuffix = std::string(argv[1]);
  std::string fieldName = std::string(argv[2]);

  // Data resolution
  int dimx = std::stoi(argv[3]);
  int dimy = std::stoi(argv[4]);
  int dimz = std::stoi(argv[5]);

  // The number of ensemble members
  int numEnsembles = std::stoi(argv[6]);
    
  // Isovalue
  double isovalue = std::stod(argv[7]);

  // Update the spacing and origin as needed by the dataset
  const viskores::Id3 dims(dimx, dimy, dimz);
  const viskores::Id3 origin(0, 0, 0);
  const viskores::Id3 spacing(2, 2, 2);
  viskores::cont::DataSetBuilderUniform dataSetBuilder;
  viskores::cont::DataSet vtkmDataSet = dataSetBuilder.Create(dims,origin,spacing);

  // Read all ensemble data
  viskores::cont::ArrayHandleRuntimeVec<viskores::FloatDefault> allEnsemblesArray(numEnsembles);
  allEnsemblesArray.Allocate(dimx * dimy * dimz);

  std::vector<viskores::cont::ArrayHandle<viskores::FloatDefault>> dataArray;
  for (int ensId = 0; ensId < numEnsembles; ensId++)
  {
      std::string fileName = dataPathSuffix + std::to_string(ensId) + ".vtk";
      viskores::io::VTKDataSetReader reader(fileName);
      viskores::cont::DataSet inData = reader.ReadDataSet();

      viskores::cont::ArrayHandle<viskores::FloatDefault> fieldDataArray;
      viskores::cont::ArrayCopyShallowIfPossible(inData.GetField(fieldName).GetData(), fieldDataArray);
      dataArray.push_back(fieldDataArray);
      //printSummary_ArrayHandle(fieldDataArray, std::cout, true);

  }

  std::cout << "ok to load the data at the first step" << std::endl;

  // Using all ensemble data
  viskores::cont::ArrayHandleRuntimeVec<viskores::FloatDefault> runtimeVecArray(numEnsembles);
  runtimeVecArray.Allocate(dimx * dimy * dimz);
  auto writePortal = runtimeVecArray.WritePortal();
  for (int k = 0; k < dimz; k++)
  {
      for (int j = 0; j < dimy; j++)
      {
         for (int i = 0; i < dimx; i++)
         {
            int pointIndex = k * dimx * dimy + j * dimx + i;
            auto vecValue = writePortal.Get(pointIndex);
            // load ensemble data from ens 0 to ens with id usedEnsembles-1
            for (int currEndId = 0; currEndId < numEnsembles; currEndId++)
            {
                // set the ensemble value
                vecValue[currEndId] = dataArray[currEndId].ReadPortal().Get(pointIndex);
            }
         }
      }
  }

  vtkmDataSet.AddPointField("ensemble", runtimeVecArray);
    
  // Run the KDE MAGIC filter and measure time
  KDE_MAGIC contourFilter;
  contourFilter.SetEnsembleField("ensemble");
  contourFilter.SetIsoValue(isovalue);
  contourFilter.SetMergeDuplicatePoints(true);
  contourFilter.SetAddInterpolationEdgeIds(true);
  timer.Start();
  viskores::cont::DataSet contour = contourFilter.Execute(vtkmDataSet);
  timer.Stop();
  std::cout << "Running time: " << timer.GetElapsedTime() << std::endl;
    
  // Write out the result for visualization in ParaView or other software
  viskores::io::VTKDataSetWriter writer("output.vtk");
  writer.SetFileTypeToBinary();
  writer.WriteDataSet(contour);

  return 0;
}
