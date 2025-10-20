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

#ifndef UCV_ENTROPY_ADAPTIVE_EIGENS_h
#define UCV_ENTROPY_ADAPTIVE_EIGENS_h

#include <viskores/worklet/WorkletMapTopology.h>
#include <cmath>
#include "linalg/EasyLinAlg/eigen.h"

#if defined(VISKORES_CUDA) || defined(VISKORES_KOKKOS_HIP)
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/normal_distribution.h>
#else
#include <random>
#endif // VISKORES_CUDA


/* Worklet to compute the mean of the ensemble.
"InPointFieldVecEnsemble" (input): ensemble,
"InPointFieldVecMean" (input): Mean of the ensemble,
"outCellFieldCProb" (output): Cell-crossing probability,
"outCellFieldNumNonzeroProb" (output): Number of non-zero probability topology cases per cell,
"outCellFieldEntropy" (output): Cell-crossing entropy.
 */
template <int NumCellCorners, int NumCases>
class EntropyAdaptiveEigens : public viskores::worklet::WorkletVisitCellsWithPoints
{
public:
    EntropyAdaptiveEigens(double isovalue, int numSamples, double thresholdInd, double thresholdEnergy)
        : m_isovalue(isovalue), m_numSamples(numSamples), m_thresholdInd(thresholdInd), m_thresholdEnergy(thresholdEnergy){};

    using ControlSignature = void(CellSetIn,
                                  FieldInPoint,
                                  FieldInPoint,
                                  FieldOutCell,
                                  FieldOutCell,
                                  FieldOutCell);

    using ExecutionSignature = void(_2, _3, _4, _5, _6);
    //  the first parameter is binded with the worklet
    using InputDomain = _1;

    template <typename InPointFieldVecEnsemble,
              typename InPointFieldVecMean,
              typename OutCellFieldType1,
              typename OutCellFieldType2,
              typename OutCellFieldType3>

    VISKORES_EXEC void operator()(
        const InPointFieldVecEnsemble &inPointFieldVecEnsemble,
        const InPointFieldVecMean &inMeanArray,
        OutCellFieldType1 &outCellFieldCProb,
        OutCellFieldType2 &outCellFieldNumNonzeroProb,
        OutCellFieldType3 &outCellFieldEntropy) const
    {
        
        viskores::IdComponent numVertices = inPointFieldVecEnsemble.GetNumberOfComponents();
        
        // Only support for 2D/3D ensemble data
        if (inMeanArray.GetNumberOfComponents() != NumCellCorners)
        {
            printf("inMeanArray in 2D version expect 4 vertices, and 3D version expects 8 vertices\n");
            return;
        }

        // Filter out cells that do not contain the isovalue
        // For the same, find the minimum and maximum value per cell data values
        using VecType = decltype(inPointFieldVecEnsemble[0]);
        double cellMin = viskores::Infinity64();
        double cellMax = viskores::NegativeInfinity64();
        for (int i = 0; i < NumCellCorners; i++)
        {
            find_min_max<VecType>(inPointFieldVecEnsemble[i], cellMin, cellMax);
        }

        // Check if the isovalue lies in the data range for a grid cell
        if (this->m_isovalue < cellMin || this->m_isovalue > cellMax)
        {
            outCellFieldCProb = 0;
            return;
        }

        // Compute the covariance matrix (8x8 for 3D data) for the data at grid cell vertices
        constexpr viskores::IdComponent covMatrixSize = (NumCellCorners + 1) * NumCellCorners / 2;
        viskores::Vec<viskores::FloatDefault, covMatrixSize> cov_matrix;
        viskores::IdComponent index = 0;
        for (int p = 0; p < numVertices; ++p)
        {
            for (int q = p; q < numVertices; ++q)
            {
                float cov = find_covariance<VecType>(inPointFieldVecEnsemble[p], inPointFieldVecEnsemble[q], inMeanArray[p], inMeanArray[q]);
                cov_matrix[index] = cov;
                index++;
            }
        }

        // Compute the mean of the ensemble at cell corners
        EASYLINALG::Vec<double, NumCellCorners> ucvmeanv;

        for (int i = 0; i < NumCellCorners; i++)
        {
            ucvmeanv[i] = inMeanArray[i];
        }

        // Covariance matrix
        int covindex = 0;
        EASYLINALG::Matrix<double, NumCellCorners, NumCellCorners> ucvcov;

        for (int p = 0; p < NumCellCorners; ++p)
        {
            for (int q = p; q < NumCellCorners; ++q)
            {
                // use the elements at the top half
                // printf("%f ", cov_matrix[covindex]);
                ucvcov[p][q] = cov_matrix[covindex];
                if (p != q)
                {
                    // assign value to another helf
                    ucvcov[q][p] = ucvcov[p][q];
                }
                covindex++;
            }
        }

        // Set the Monte Carlo sample count
        viskores::IdComponent numSamples = this->m_numSamples;

        //  Transform the isovalue for the eigenvalue-driven approach. Subtract mean of the ensemble data from isovalue.
        EASYLINALG::Vec<double, NumCellCorners> transformIso(0);
        for (int i = 0; i < NumCellCorners; i++)
        {
            transformIso[i] = this->m_isovalue - ucvmeanv[i];
        }

        // Compute eigen values of the covariance matrix
        EASYLINALG::Vec<double, NumCellCorners> eigenValues;
        EASYLINALG::SymmEigenValues(ucvcov, this->m_tolerance, this->m_iterations, eigenValues);
        using EigenValuesType = decltype(eigenValues);

        // Sort the eigen values in a decending order
        Sort<EigenValuesType>(eigenValues);

        // Make sure eigen values are in descending order
        if (checkOrder<EigenValuesType>(eigenValues) == false)
        {
            // print out
            printf("wrong eigne value sequence\n");
            eigenValues.Show();
        }

        // Choosing the number of important eigen values
        EASYLINALG::Vec<double, NumCellCorners> eigenValuesFiltered(0);
        
        // Using the first eigen value as a default case
        int filteredEigenCount = 1;
        eigenValuesFiltered[0] = eigenValues[0];
        
        // Filter out the eigen values when it is less then the threshold
        // and not all eigen values/dimensions of the covariance matrix are used for Monte Carlo sampling.
        // So sampling will be performed in dimension smaller than usual 8-dimensional space for a 3D grid cell.
        // Assuming this->m_thresholdEnergy is larger than 0
        for (int i = 1; i < NumCellCorners; i++)
        {
            // Keep dimensions that have eigenvalue greater than the fraction (user-specified) of the highest eigenvalue
            if (eigenValues[i] > this->m_thresholdEnergy * eigenValues[0])
            {
                eigenValuesFiltered[filteredEigenCount] = eigenValues[i];
                filteredEigenCount++;
            }
        }

        // Compute eigen vectors only for important eigen values (this number will likely be smaller than 8 for 3D data, thereby helping to reduce the amount of sampling)
        EASYLINALG::Vec<EASYLINALG::Vec<double, NumCellCorners>, NumCellCorners> eigenVectors;
        // how many eigen vector we want to use

        for (int i = 0; i < filteredEigenCount; i++)
        {
            eigenVectors[i] = EASYLINALG::ComputeEigenVectors(ucvcov, eigenValuesFiltered[i], this->m_iterations);
        }

        viskores::Vec<viskores::FloatDefault, NumCases> probHistogram;

        EASYLINALG::Vec<double, NumCellCorners> sample_v;

#if defined(VTKM_CUDA) || defined(VTKM_KOKKOS_HIP)
        thrust::minstd_rand rng;
        thrust::random::normal_distribution<double> norm(0, 1);
#else
        std::mt19937 rng;
        rng.seed(std::mt19937::default_seed);
        std::normal_distribution<double> norm(0, 1);
#endif // VTKM_CUDA

        // An Array to store the histogram of marching cubes topology cases
        // There are 2^8 = 256 topological configurations for 3D data
        for (int i = 0; i < NumCases; i++)
        {
            probHistogram[i] = 0.0;
        }

        // Generate low-dimensional samples (determined based on eigenvalues/eigenvectors as above)
        for (viskores::Id n = 0; n < numSamples; ++n)
        {
            EASYLINALG::Vec<double, NumCellCorners> sampleResults(0);

            for (int i = 0; i < filteredEigenCount; i++)
            {

                // vtkm will return nan for negative sqrt value
                // just filter it out in the previous step when filter
                // the eigen value
                sample_v[i] = viskores::Sqrt(eigenValues[i]) * norm(rng);
            }

            for (int i = 0; i < NumCellCorners; i++)
            {
                for (int j = 0; j < filteredEigenCount; j++)
                {
                    // eigen vector of jth eigen value, jth sample element
                    // ith componnet in the eigen vector
                    sampleResults[i] += eigenVectors[j][i] * sample_v[j];
                }
            }

            // Check which marching cubes topological case the sample belongs to and
            // increment the count for that specific topological case in the array to
            // store the topology case histogram
            uint caseValue = 0;
            for (uint i = 0; i < NumCellCorners; i++)
            {
                // setting associated position to 1 if the isovalue is larger than sample value at a grid vertex
                if (transformIso[i] >= sampleResults[i])
                {
                    caseValue = (1 << i) | caseValue;
                }
            }

            // increment
            probHistogram[caseValue] = probHistogram[caseValue] + 1.0;
        }

        // Normalize the histogram
        for (int i = 0; i < NumCases; i++)
        {
            probHistogram[i] = (probHistogram[i] / (1.0 * numSamples));
        }

        // Cell-crossing probability =  1 - probabilities for the non-crossing topological cases
        outCellFieldCProb = 1.0 - (probHistogram[0] + probHistogram[NumCases - 1]);

        viskores::Id nonzeroCases = 0;
        viskores::FloatDefault entropyValue = 0;
        viskores::FloatDefault templog = 0;
        
        // Compute the number of topological cases with non-zero probability and
        // compute the per-cell topological entropy
        for (int i = 0; i < NumCases; i++)
        {
            if (probHistogram[i] > 0.0001)
            {
                nonzeroCases++;
                templog = viskores::Log2(probHistogram[i]);
            }
            // do not update entropy if the pro is zero
            entropyValue = entropyValue + (-probHistogram[i]) * templog;
        }

        // Return the non-zero probability topological case counts and topological entropy results.
        outCellFieldNumNonzeroProb = nonzeroCases;
        outCellFieldEntropy = entropyValue;
    }

    // Routine to sort 1D array
    template <typename VecType>
    VISKORES_EXEC inline void Sort(VecType &arr) const
    {
        viskores::Id num = arr.NUM_COMPONENTS;
        for (int i = 0; i < num; i++)
        {
            for (int j = 0; j < num - i - 1; j++)
            {
                // compare element i and j
                if (arr[j] < arr[j + 1])
                {
                    // swap
                    auto temp = arr[j];
                    arr[j] = arr[j + 1];
                    arr[j + 1] = temp;
                }
            }
        }
        return;
    }

    // Routine to check if array elements are ordered in a decending order
    template <typename VecType>
    VISKORES_EXEC bool checkOrder(const VecType &eigenValues) const
    {
        viskores::Id num = eigenValues.NUM_COMPONENTS;
        for (viskores::Id i = 0; i < num - 1; i++)
        {
            if (eigenValues[i] < eigenValues[i + 1])
            {
                // in ascending order
                return false;
            }
        }
        return true;
    }

    // Routine to find a minimum and maximum of an array
    template <typename VecType>
    VISKORES_EXEC void find_min_max(const VecType &arr, viskores::Float64 &min, viskores::Float64 &max) const
    {
        viskores::Id num = arr.GetNumberOfComponents();
        for (viskores::Id i = 0; i < num; i++)
        {
            // the second one is runtime thing (recombine vec), so convert it into float defualt firt
            viskores::Float64 v = arr[i];
            min = viskores::Min(min, v);
            max = viskores::Max(max, v);
        }
        return;
    }

    // Routine to compute the covariance of two 1D arrays with them and their means as inputs
    template <typename VecType>
    VISKORES_EXEC inline viskores::FloatDefault find_covariance(const VecType &arr1, const VecType &arr2,
                                                        const viskores::FloatDefault &mean1, const viskores::FloatDefault &mean2) const
    {
        if (arr1.GetNumberOfComponents() != arr2.GetNumberOfComponents())
        {
            printf("error, failed to compute find_covariance, the array size should be equal with each other\n");
            return 0;
        }
        viskores::Id arraySize = arr1.GetNumberOfComponents();
        viskores::FloatDefault sum = 0;
        for (int i = 0; i < arraySize; i++)
        {
            viskores::FloatDefault v1 = arr1[i];
            viskores::FloatDefault v2 = arr2[i];
            sum = sum + (v1 - mean1) * (v2 - mean2);
        }

        return sum / (viskores::FloatDefault)(arraySize - 1);
    }

private:
    double m_isovalue;
    int m_numSamples;
    int m_iterations = 200;
    double m_tolerance = 0.0001;

    // threshold to depend if there is sphere covaraince structure
    double m_thresholdInd = 1.0;
    // threshold to determine the number of eigen values we should keep
    double m_thresholdEnergy = 0.1;
};

#endif // UCV_MULTIVARIANT_GAUSSIAN3D_h
