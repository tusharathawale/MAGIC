/*============================================================================

Title   -  MAGIC: Marching Cubes Isosurface Uncertainty Visualization for Gaussian Uncertain Data with Spatial Correlation
Authors -  Tushar M. Athawale, Kenneth Moreland, David Pugmire, Chris R. Johnson, Paul Rosen, Matthew Norman, Antigoni Georgiadou, and Alireza Entezari
Date    -  Oct 20, 2025
 
This code implements the following three previous papers using the Viskores library:

(1) Z. Wang, T. M. Athawale, K. Moreland, J. Chen, C. R. Johnson, and
 D. Pugmire, “FunMC^2: A Filter for Uncertainty Visualization of
 Marching Cubes on Multi-Core Devices,” in Eurographics Symposium
 on Parallel Graphics and Visualization, R. Bujack, D. Pugmire, and
 G. Reina, Eds. The Eurographics Association, 2023.
 doi: 10.2312/pgv.20231081
 
(2) T. M. Athawale, S. Sane and C. R. Johnson, "Uncertainty Visualization
 of the Marching Squares and Marching Cubes Topology Cases," in 2021 IEEE
 Visualization Conference (VIS), New Orleans, LA, USA, 2021, pp. 106-110,
 doi: 10.1109/VIS49827.2021.9623267.
 
(3) K. P¨othkow, B. Weber, and H.-C. Hege, “Probabilistic marching cubes,”
 Computer Graphics Forum, vol. 30, no. 3, pp. 931–940, 2011.
 doi: 10.1111/j.1467-8659.2011.01942.x
 
This code is distributed under the OSI-approved BSD 3-clause License. See LICENSE.txt for details.
 
============================================================================*/

#ifndef UCV_MULTIVARIANT_GAUSSIAN3D2_h
#define UCV_MULTIVARIANT_GAUSSIAN3D2_h

#include <viskores/worklet/WorkletMapTopology.h>
#include <cmath>

#include "./linalg/ucv_matrix_static_8by8.h"


/* Worklet to compute the cell crossing probability/entropy for an isosurface.
"inPointFieldVecEnsemble" (input): Ensemble data,
"outCellFieldCProb" (output): Cell-crossing probability,
"outCellFieldNumNonzeroProb" (output): Number of non-zero probability topology cases per cell,
"outCellFieldEntropy" (output): Cell-crossing entropy,
"randomNumbers" (input): Random numbers generated with fixed seed.
 */
class MVGaussianTemp : public viskores::worklet::WorkletVisitCellsWithPoints
{
public:
MVGaussianTemp(double isovalue, viskores::Id num_sample)
        : m_isovalue(isovalue), m_num_sample(num_sample){};

    using ControlSignature = void(CellSetIn,
                                  FieldInPoint,
                                  FieldOutCell,
                                  FieldOutCell,
                                  FieldOutCell,
                                  WholeArrayIn);

    using ExecutionSignature = void(_2, _3, _4, _5, _6);

    // the first parameter (cell set) is binded with the worklet
    using InputDomain = _1;
    // InPointFieldType should be a vector
    template <typename InPointFieldVecEnsemble,
              typename OutCellFieldType1,
              typename OutCellFieldType2,
              typename OutCellFieldType3,
              typename RandomPortalType>

    VISKORES_EXEC void operator()(
        const InPointFieldVecEnsemble &inPointFieldVecEnsemble,
        OutCellFieldType1 &outCellFieldCProb,
        OutCellFieldType2 &outCellFieldNumNonzeroProb,
        OutCellFieldType3 &outCellFieldEntropy,
        const RandomPortalType &randomNumbers) const
    {
    
        viskores::IdComponent numVertices = inPointFieldVecEnsemble.GetNumberOfComponents();

        // Using numVertices to decide the length of mean and cov
        // and decide them at the runtime. Currently, worklet is designed for a
        // 3D dataset. So the number of vertices is 8 per 3D cell.
        if (numVertices != 8)
        {
            printf("The cell crossing probability/entropy worklet only supports cell with 8 vertices");
            return;
        }

        // get the type in the fieldVec
        // the VecType specifies the number of ensembles
        using VecType = decltype(inPointFieldVecEnsemble[0]);
                 
        UCVMATH::vec_t ucvmeanv;

        // Compute the mean at cell vertices.
        for (int i = 0; i < numVertices; i++)
        {
             ucvmeanv.v[i] = find_mean<VecType>(inPointFieldVecEnsemble[i]);
        }

        // Set the trim options to filter the 0 values.
        if (fabs(ucvmeanv.v[0]) < 0.000001 && fabs(ucvmeanv.v[1]) < 0.000001 && fabs(ucvmeanv.v[2]) < 0.000001 && fabs(ucvmeanv.v[3]) < 0.000001 && fabs(ucvmeanv.v[4]) < 0.000001 && fabs(ucvmeanv.v[5]) < 0.000001 && fabs(ucvmeanv.v[6]) < 0.000001 && fabs(ucvmeanv.v[7]) < 0.000001)
        {
            outCellFieldCProb = 0;
            return;
        }

        // Compute the covariance matrix.
        // For an 8*8 covariance matrix, there are 36 numbers at the upper corner above the diagonal entries.
        viskores::Vec<viskores::FloatDefault, 36> cov_matrix;
        viskores::IdComponent index = 0;
        for (int p = 0; p < numVertices; ++p)
        {
            for (int q = p; q < numVertices; ++q)
            {
                float cov = find_covariance<VecType>(inPointFieldVecEnsemble[p], inPointFieldVecEnsemble[q], ucvmeanv.v[p], ucvmeanv.v[q]);
                cov_matrix[index] = cov;
                index++;
            }
        }

        UCVMATH::mat_t ucvcov8by8;
        int covindex = 0;
        for (int p = 0; p < numVertices; ++p)
        {
            for (int q = p; q < numVertices; ++q)
            {
                // use the elements at the top half
                // printf("%f ", cov_matrix[covindex]);
                ucvcov8by8.v[p][q] = cov_matrix[covindex];
                if (p != q)
                {
                    // assign value to another helf
                    ucvcov8by8.v[q][p] = ucvcov8by8.v[p][q];
                }
                covindex++;
            }
        }

        // Eigen value decomposition of the covariance matrix to generate a sample of
        // multivariate Gauassian (here, 8-variate distribution)
        UCVMATH::mat_t A = UCVMATH::eigen_vector_decomposition(&ucvcov8by8);
        
        // Arrays to store sample-relevant data
        UCVMATH::vec_t sample_v;
        UCVMATH::vec_t AUM;

        // Set the Monte Carlo sample count.
        viskores::IdComponent numSamples = m_num_sample;
        
        // An Array to store the histogram of marching cubes topology cases
        // There are 2^8 = 256 topological configurations
        viskores::Vec<viskores::FloatDefault, 256> probHistogram;
        for (int i = 0; i < 256; i++)
        {
            probHistogram[i] = 0.0;
        }

        // Generate samples by multiplying eigen decomposition (computed above) to the random array
        // followed with addition of the mean (computed above)
        for (viskores::Id n = 0; n < numSamples; ++n)
        {
            // get random numbers at eight corners
            for (int i = 0; i < numVertices; i++)
            {
                // using other sample mechanism such as thrust as needed
                VISKORES_ASSERT((i + (n * numVertices)) < randomNumbers.GetNumberOfValues());
                sample_v.v[i] = randomNumbers.Get(i + (n * numVertices));
            }

            // Multiply by eigenvectors and add mean
            AUM = UCVMATH::matrix_mul_vec_add_vec(&A, &sample_v, &ucvmeanv);

            // Check which marching cubes topological case the sample belongs to and
            // increment the count for that specific topological case in the array to
            // store the topology case histogram
            uint caseValue = 0;
            for (viskores::IdComponent i = 0; i < numVertices; i++)
            {
                // setting associated position to 1 if iso larger then specific cases
                if (m_isovalue >= AUM.v[i])
                {
                    caseValue = (1 << i) | caseValue;
                }
            }

            // increment
            probHistogram[caseValue] = probHistogram[caseValue] + 1.0;
        }

        // Normalize the histogram
        for (int i = 0; i < 256; i++)
        {
             probHistogram[i] = (probHistogram[i] / (1.0 * numSamples));
        }

        // Cell-crossing probability =  1 - probabilities for the non-crossing topological cases
        // (i.e., the cases 0 and 255).
        // Return the cell-crossing probability result.
        outCellFieldCProb = 1.0 - (probHistogram[0] + probHistogram[255]);

        viskores::Id nonzeroCases = 0;
        viskores::FloatDefault entropyValue = 0;
        viskores::FloatDefault templog = 0;
        
        // Compute the number of topological cases with non-zero probability and
        // compute the per-cell topological entropy
        for (int i = 0; i < 256; i++)
        {
            if (probHistogram[i] > 0.0001)
            {
                nonzeroCases++;
                templog = viskores::Log2(probHistogram[i]);
            }
            entropyValue = entropyValue + (-probHistogram[i]) * templog;
        }

        // Return the non-zero probability topologica case counts and topological entropy results.
        outCellFieldNumNonzeroProb = nonzeroCases;
        outCellFieldEntropy = entropyValue;
    }

    // Rountine to compute the mean of the input ensemble dataset
    template <typename VecType>
    VISKORES_EXEC viskores::Float64 find_mean(const VecType &arr) const
    {
        viskores::Float64 sum = 0;
        viskores::Id num = arr.GetNumberOfComponents();
        for (viskores::Id i = 0; i < arr.GetNumberOfComponents(); i++)
        {
            sum = sum + arr[i];
        }
        viskores::Float64 mean = (1.0 * sum) / (1.0 * num);
        return mean;
    }
    
    // Routine to compute the covariance of two 1D arrays with them and their means as inputs
    template <typename VecType>
    VISKORES_EXEC double find_covariance(const VecType &arr1, const VecType &arr2,
                                     double &mean1, double &mean2) const
    {
        if (arr1.GetNumberOfComponents() != arr2.GetNumberOfComponents())
        {
            // cuda does not support exception
            printf("error, failed to compute find_covariance, the array size should be equal with each other\n");
            return 0;
        }
        viskores::Id arraySize = arr1.GetNumberOfComponents();
        double sum = 0;
        for (int i = 0; i < arraySize; i++)
            sum = sum + (arr1[i] - mean1) * (arr2[i] - mean2);
        return (double)sum / (double)(arraySize - 1);
    }

private:
    double m_isovalue;
    int m_num_sample = 1000;
};

#endif // UCV_MULTIVARIANT_GAUSSIAN2D_h
