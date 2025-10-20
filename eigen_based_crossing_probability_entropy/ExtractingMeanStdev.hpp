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

#ifndef UCV_EXTRACTING_MEAN_STD_h
#define UCV_EXTRACTING_MEAN_STD_h

#include <viskores/worklet/WorkletMapField.h>
#include <viskores/worklet/WorkletReduceByKey.h>
#include <cmath>

/* Worklet to compute the mean of the ensemble.
"inPointFieldVecEnsemble" (input): ensemble data,
"meanValue" (output): Mean of the ensemble.
 */
struct ExtractingMean : public viskores::worklet::WorkletMapField
{
    using ControlSignature = void(FieldIn, FieldOut);
    using ExecutionSignature = void(_1, _2);
    using InputDomain = _1;
    template <typename OriginalValuesType, typename OutputType>
    VISKORES_EXEC void operator()(
        const OriginalValuesType &inPointFieldVecEnsemble, OutputType &meanValue) const
    {
        viskores::FloatDefault boxSum = 0;

        viskores::IdComponent NumComponents = inPointFieldVecEnsemble.GetNumberOfComponents();

        // Compute the mean.
        for (viskores::IdComponent index = 0;
             index < NumComponents; index++)
        {
            boxSum = boxSum + static_cast<viskores::FloatDefault>(inPointFieldVecEnsemble[index]);
        }

        meanValue = boxSum / (1.0 * (NumComponents));
    }
};

/* Worklet to compute the mean and standard deviation of the ensemble.
"inPointFieldVecEnsemble" (input): ensemble data,
"meanValue" (output): Mean of the ensemble.
"stdevValue" (output): Standard deviation of the ensemble.
 */
struct ExtractingMeanStdevEnsembles : public viskores::worklet::WorkletMapField
{
    using ControlSignature = void(FieldIn, FieldOut, FieldOut);
    using ExecutionSignature = void(_1, _2, _3);
    using InputDomain = _1;
    template <typename OriginalValuesType, typename OutputType>
    VISKORES_EXEC void operator()(
        const OriginalValuesType &inPointFieldVecEnsemble, OutputType &meanValue, OutputType &stdevValue) const
    {
        viskores::FloatDefault boxSum = 0;

        viskores::IdComponent NumComponents = inPointFieldVecEnsemble.GetNumberOfComponents();

        // Compute the mean.
        for (viskores::IdComponent index = 0;
             index < NumComponents; index++)
        {
            boxSum = boxSum + static_cast<viskores::FloatDefault>(inPointFieldVecEnsemble[index]);
        }

        meanValue = boxSum / (1.0 * (NumComponents));

        // Compute the standard deviation
        viskores::FloatDefault diffSum = 0;
        for (viskores::IdComponent index = 0;
             index < NumComponents; index++)
        {
            viskores::FloatDefault diff = static_cast<viskores::FloatDefault>(inPointFieldVecEnsemble[index]) - static_cast<viskores::FloatDefault>(meanValue);
            diffSum += diff * diff;
        }

        stdevValue = std::sqrt(diffSum / (1.0*NumComponents));
    }
};

/* Worklet to compute the mean and standard deviation of the ensemble.
"originalValues" (input): ensemble data,
"meanValue" (output): Mean of the ensemble.
"stdevValue" (output): Standard deviation of the ensemble.
 */
struct ExtractingMeanStdev : public viskores::worklet::WorkletReduceByKey
{
    using ControlSignature = void(KeysIn, ValuesIn, ReducedValuesOut, ReducedValuesOut);
    using ExecutionSignature = void(_2, _3, _4);
    using InputDomain = _1;
    template <typename OriginalValuesType, typename OutputType>
    VISKORES_EXEC void operator()(
        const OriginalValuesType &originalValues, OutputType &meanValue, OutputType &stdevValue) const
    {
        viskores::FloatDefault boxSum = 0;

        viskores::IdComponent NumComponents = originalValues.GetNumberOfComponents();

        // Compute the mean.
        for (viskores::IdComponent index = 0;
             index < NumComponents; index++)
        {
            boxSum = boxSum + static_cast<viskores::FloatDefault>(originalValues[index]);
        }

        meanValue = boxSum / (1.0 * (NumComponents));

        // Compute the standard deviation
        viskores::FloatDefault diffSum = 0;
        for (viskores::IdComponent index = 0;
             index < NumComponents; index++)
        {
            viskores::FloatDefault diff = static_cast<viskores::FloatDefault>(originalValues[index]) - static_cast<viskores::FloatDefault>(meanValue);
            diffSum += diff * diff;
        }

        stdevValue = std::sqrt(diffSum / (1.0*NumComponents));
    }
};

#endif // UCV_EXTRACTING_MEAD_STD_h
