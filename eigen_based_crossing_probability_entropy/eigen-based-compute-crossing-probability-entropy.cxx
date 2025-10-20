/*============================================================================

Title   -  MAGIC: Marching Cubes Isosurface Uncertainty Visualization for Gaussian Uncertain Data with Spatial Correlation
Authors -  Tushar M. Athawale, Kenneth Moreland, David Pugmire, Chris R. Johnson, Paul Rosen, Matthew Norman, Antigoni Georgiadou, and Alireza Entezari
Date    -  Oct 20, 2025
 
This code is a courtesy of the following paper:

 T. M. Athawale, Z. Wang, C. R. Johnson, and D. Pugmire, “Data-Driven
 Computation of Probabilistic Marching Cubes for Efficient Visualization
 of Level-Set Uncertainty,” in EuroVis (Short Papers), C. Tominski,
 M. Waldner, and B. Wang, Eds. The Eurographics Association, 2024.
 doi: 10.2312/evs.20241071
 
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
#include <viskores/cont/ArrayHandleRuntimeVec.h>
#include <viskores/cont/ArrayCopy.h>
#include <viskores/cont/ArrayHandleRuntimeVec.h>
#include <viskores/cont/Timer.h>
#include <viskores/cont/DataSetBuilderUniform.h>

#include <viskores/worklet/WorkletMapField.h>
#include <math.h>
#include <float.h>

#include "ExtractingMeanStdev.hpp"
#include "EntropyAdaptiveEigens.hpp"


// Update the number of components based on the number of ensemble members in the data
using SupportedTypesVec = viskores::List<viskores::Vec<float, 6>>;

/* Function to call probabilistic marching cubes worklet for a 3D grid cell.
"vtkmDataSet" (input): Array handle representing the ensemble data,
"isovalue" (input): Isovalue,
"numSamples" (input): Number of Monte Carlo samples,
"outputFileNameSuffix" (input): For output file name,
"use2d" (input): Is the data 2D or not?,
"eigenThreshold" (input): Minimum eigen value needed to consider dimension for sampling of a multivariate Gaussian,
"timer" (input): Measure time,
"writeFile" (input): Write a file or not?,
"(output): Array handle containing the computed cell-crossing probability and per-cell isosurface entropy.
 */
viskores::cont::ArrayHandle<viskores::Float64> ComputeEntropyWithRuntimeVec(viskores::cont::DataSet vtkmDataSet,
                                                                    double isovalue, int numSamples, std::string outputFileNameSuffix, bool use2d, double eigenThreshold, viskores::cont::Timer &timer, std::string writeFile)
{
    timer.Start();
    // Processing current ensemble data sets based on the uncertainty countour
    viskores::cont::ArrayHandle<viskores::FloatDefault> meanArray;
    viskores::cont::ArrayHandle<viskores::Float64> crossProbability;
    viskores::cont::ArrayHandle<viskores::Id> numNonZeroProb;
    viskores::cont::ArrayHandle<viskores::Float64> entropy;
    auto resolveType = [&](auto &concreteArray)
    {
        viskores::cont::Invoker invoke;
        viskores::Id numPoints = concreteArray.GetNumberOfValues();
        auto concreteArrayView = viskores::cont::make_ArrayHandleView(concreteArray, 0, numPoints);

        // Worklet to compute the mean of the ensemble data
        invoke(ExtractingMean{}, concreteArrayView, meanArray);
        // printSummary_ArrayHandle(meanArray, std::cout);
        // printSummary_ArrayHandle(stdevArray, std::cout);
        
        // Check if the data is 2D or 3D
        if (use2d)
        {
            // Worklet to compute the eigenvalue-driven cell-crossing probability/entropy for an ensemble of 2D data
            invoke(EntropyAdaptiveEigens<4, 16>{isovalue, numSamples, 1.0, eigenThreshold},
                   vtkmDataSet.GetCellSet(),
                   concreteArrayView,
                   meanArray,
                   crossProbability,
                   numNonZeroProb,
                   entropy);
        }
        else
        {
            // Worklet to compute the eigenvalue-driven cell-crossing probability/entropy for an ensemble of 3D data
            invoke(EntropyAdaptiveEigens<8, 256>{isovalue, numSamples, 1.0, eigenThreshold},
                   vtkmDataSet.GetCellSet(),
                   concreteArrayView,
                   meanArray,
                   crossProbability,
                   numNonZeroProb,
                   entropy);
        }

        // Output dataset containing cell-crossing probability, per-cell number of cases with nonzero probability, and per-cell entropy
        auto outputDataSet = vtkmDataSet;
        
        timer.Stop();
        std::cout << "worklet time is:" << timer.GetElapsedTime() << std::endl;

        std::stringstream stream;
        stream << isovalue;
        std::string isostr = stream.str();

        // Create result data with the computed cell-crossing and entropy fields
        std::string outputFileName = outputFileNameSuffix + isostr + ".vtk";
        outputDataSet.AddCellField("cross_prob", crossProbability);
        outputDataSet.AddCellField("num_nonzero_prob", numNonZeroProb);
        outputDataSet.AddCellField("entropy", entropy);
        if (writeFile == "true")
        {
            viskores::io::VTKDataSetWriter writeCross(outputFileName);
            writeCross.WriteDataSet(outputDataSet);
        }
    };
    
    // Call the worklet by resolving the type of the data for ensembles field
    vtkmDataSet.GetField("ensembles")
        .GetData()
        .CastAndCallWithExtractedArray(resolveType);

    return crossProbability;
}

int main(int argc, char *argv[])
{
    // Initialize the viskores device and timer
    viskores::cont::InitializeResult initResult = viskores::cont::Initialize(
        argc, argv, viskores::cont::InitializeOptions::DefaultAnyDevice);
    viskores::cont::Timer timer{initResult.Device};

    // Template to run the crossing probability and entropy filter
    // Example: ./eigen-based-crossing-probability file_path/tangle_correlation_one_ tangle 64 64 64 6 100 27.6 tangle_cross_prob 1 true
    if (argc != 12)
    {
        //eigenThreshold represent a fraction of the maximum eigen value for the covariance matrix
        std::cout << "<executable> <SyntheticDataSuffix> <FieldName> <Dimx> <Dimy> <Dimz> <num of ensembles> <num of samples> <isovalue> <outputFileSuffix> <eigenThreshold> <true/false (write file or not)>" << std::endl;
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

    // The number of ensemble members and Monte Carlo samples
    int numEnsembles = std::stoi(argv[6]);
    int numSamples = std::stoi(argv[7]);
    
    // Isovalue
    double isovalue = std::stod(argv[8]);
    
    std::string outputSuffix = std::string(argv[9]);
    double eigenThreshold = std::stod(argv[10]);
    std::string writeFile = std::string(argv[11]);
    
    //eigenThreshold represent a fraction of the maximum eigen value for the covariance matrix
    if (eigenThreshold < 0)
    {
        throw std::runtime_error("eigenThreshold is supposed to be larger than 0");
    }
    
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
    // redsea data start from 1
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
    //printSummary_ArrayHandle(runtimeVecArray, std::cout);
    
    // Call worklet for 3D data
    bool use2d = false;
    if (dimz == 1)
    {
        use2d = true;
    }

    std::cout << "start to call the adaptive worklet" << std::endl;
    auto crossProb1 = ComputeEntropyWithRuntimeVec(vtkmDataSet, isovalue, numSamples, outputSuffix, use2d, eigenThreshold, timer, writeFile);
    
    return 0;
}
