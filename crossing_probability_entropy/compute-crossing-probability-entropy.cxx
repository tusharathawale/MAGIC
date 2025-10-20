/*============================================================================

Title   -  MAGIC: Marching Cubes Isosurface Uncertainty Visualization for Gaussian Uncertain Data with Spatial Correlation
Authors -  Tushar M. Athawale, Kenneth Moreland, David Pugmire, Chris R. Johnson, Paul Rosen, Matthew Norman, Antigoni Georgiadou, and Alireza Entezari
Date    -  Oct 20, 2025
 
This code is a courtesty of the following paper:

 Z. Wang, T. M. Athawale, K. Moreland, J. Chen, C. R. Johnson, and
 D. Pugmire, “FunMC^2: A Filter for Uncertainty Visualization of
 Marching Cubes on Multi-Core Devices,” in Eurographics Symposium
 on Parallel Graphics and Visualization, R. Bujack, D. Pugmire, and
 G. Reina, Eds. The Eurographics Association, 2023.
 doi: 10.2312/pgv.20231081
 
The code utilizes the foundational techniques proposed in the following two papers.
 
(1) T. M. Athawale, S. Sane and C. R. Johnson, "Uncertainty Visualization
 of the Marching Squares and Marching Cubes Topology Cases," in 2021 IEEE
 Visualization Conference (VIS), New Orleans, LA, USA, 2021, pp. 106-110,
 doi: 10.1109/VIS49827.2021.9623267.
 
(2) K. P¨othkow, B. Weber, and H.-C. Hege, “Probabilistic marching cubes,”
 Computer Graphics Forum, vol. 30, no. 3, pp. 931–940, 2011.
 doi: 10.1111/j.1467-8659.2011.01942.x
 
This code is distributed under the OSI-approved BSD 3-clause License. See LICENSE.txt for details.
 
============================================================================*/

#include <viskores/io/VTKDataSetWriter.h>
#include <viskores/io/VTKDataSetReader.h>

#include <viskores/cont/Initialize.h>
#include <viskores/cont/ArrayCopy.h>
#include <viskores/cont/ArrayHandleRuntimeVec.h>
#include <viskores/cont/ArrayHandleRandomStandardNormal.h>
#include <viskores/cont/Timer.h>
#include <viskores/cont/DataSetBuilderUniform.h>
#include <viskores/worklet/WorkletMapField.h>
#include <math.h>
#include <float.h>
#include "MVGaussianTemp.hpp"

// Update the number of components based on the number of ensemble members in the data.
using SupportedTypesVec = viskores::List<viskores::Vec<float, 6>>;

// Set the number of trials to run the same experiment and measure the average time.
constexpr viskores::IdComponent NUM_TRIALS = 1;

/* Function to call probabilistic marching cubes worklet for a 3D grid cell.
"vtkmDataSet" (input): Array handle representing the ensemble data,
"isovalue" (input): Isovalue,
"numSamples" (input): Number of Monte Carlo samples,
"timer" (input): Measure time,
"(output): A viskores dataset returned with the computed cell-crossing probability and per-cell isosurface entropy.
 */
viskores::cont::DataSet callWorklet(viskores::cont::DataSet& vtkmDataSet, double isovalue, viskores::Id numSamples, viskores::cont::Timer &timer)
{
    timer.Start();
    viskores::cont::ArrayHandle<viskores::Float64> crossProbability;
    viskores::cont::ArrayHandle<viskores::Id> numNonZeroProb;
    viskores::cont::ArrayHandle<viskores::Float64> entropy;
    // Set seed for a random array
    viskores::cont::ArrayHandleRandomStandardNormal<viskores::Float64> randomArray(numSamples * 8, { 5489 });
    using WorkletType = MVGaussianTemp;
    using DispatcherType = viskores::worklet::DispatcherMapTopology<WorkletType>;
    
    // Call the crossing probability worklet
    auto resolveType = [&](const auto &concrete)
    {
        DispatcherType dispatcher(MVGaussianTemp{isovalue, numSamples});
        dispatcher.Invoke(vtkmDataSet.GetCellSet(), concrete, crossProbability, numNonZeroProb, entropy, randomArray);
    };
    
    //vtkmDataSet.PrintSummary(std::cout);
    vtkmDataSet.GetField("ensembles").GetData().CastAndCallForTypes<SupportedTypesVec, VISKORES_DEFAULT_STORAGE_LIST>(resolveType);
    timer.Stop();
    std::cout << "worklet time is: " << timer.GetElapsedTime() << std::endl;

    // Create result data with the computed cell-crossing and entropy fields
    viskores::cont::DataSet outputDataSet;
    outputDataSet.CopyStructure(vtkmDataSet);
    outputDataSet.AddCellField("cross_prob", crossProbability);
    outputDataSet.AddCellField("num_nonzero_prob", numNonZeroProb);
    outputDataSet.AddCellField("entropy", entropy);

    return outputDataSet;
}

int main(int argc, char *argv[])
{
    // Initialize the viskores device and timer
    viskores::cont::InitializeResult initResult = viskores::cont::Initialize(
        argc, argv, viskores::cont::InitializeOptions::DefaultAnyDevice);
    viskores::cont::Timer timer{initResult.Device};

    // Template to run the crossing probability and entropy filter
    // Example: ./crossing-probability-entropy file_path/tangle_correlation_one_ tangle 64 64 64 14 100 27.6
    if (argc != 9)
    {
        std::cout << "<executable> <SyntheticDataSuffix> <FieldName> <Dimx> <Dimy> <Dimz> <num of ensembles> <num of samples> <isovalue>" << std::endl;
        exit(0);
    }

    std::cout << "timer device: " << timer.GetDevice().GetName() << std::endl;

    // Read all ensemble members
    std::string dataPathSuffix = std::string(argv[1]);
    std::string fieldName = std::string(argv[2]);

    // Data resolution
    int dimx = std::stoi(argv[3]);
    int dimy = std::stoi(argv[4]);
    int dimz = std::stoi(argv[5]);

    // The number of ensemble members
    int numEnsembles = std::stoi(argv[6]);
    int numSamples = std::stoi(argv[7]);
    std::cout << "num samples: " << numSamples << std::endl;
    
    // Isovalue
    double isovalue = std::stod(argv[8]);

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

    // using all ensembles
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
                //printf("debug pointIndex %d\n",pointIndex);
                auto vecValue = writePortal.Get(pointIndex);
                // load ensemble data from ens 0 to ens with id usedEnsembles-1
                for (int currEndId = 0; currEndId < numEnsembles; currEndId++)
                {
                    // set ensemble value
                    vecValue[currEndId] = dataArray[currEndId].ReadPortal().Get(pointIndex);
                }
            }
        }
    }

    vtkmDataSet.AddPointField("ensembles", runtimeVecArray);
    // printSummary_ArrayHandle(runtimeVecArray, std::cout);

    // Run the crossing probability/entropy filter
    viskores::cont::DataSet outputDataSet;
    for (viskores::IdComponent trial = 0; trial < NUM_TRIALS; ++trial)
    {
        outputDataSet = callWorklet(vtkmDataSet, isovalue, numSamples, timer);
    }

    // Write out the result for visualization in ParaView or other software
    std::string outputFileName = "./ucv_3d_iso_" + std::to_string(isovalue) + ".vtk";
    viskores::io::VTKDataSetWriter writeResult(outputFileName);
    writeResult.SetFileTypeToBinary();
    writeResult.WriteDataSet(outputDataSet);

    return 0;
}
