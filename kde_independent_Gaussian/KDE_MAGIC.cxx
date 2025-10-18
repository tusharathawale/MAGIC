/*============================================================================

Title   -  MAGIC: Marching Cubes Isosurface Uncertainty Visualization for Gaussian Uncertain Data with Spatial Correlation
Authors -  Tushar M. Athawale, Kenneth Moreland, David Pugmire, Chris R. Johnson, Paul Rosen, Matthew Norman, Antigoni Georgiadou, and Alireza Entezari
Date    -  Oct 20, 2025

This code is distributed under the OSI-approved BSD 3-clause License. See LICENSE.txt for details.
 
============================================================================*/

#include "KDE_MAGIC.h"

#include <viskores/cont/ArrayCopy.h>
#include <viskores/Math.h>
#include <viskores/worklet/WorkletMapField.h>
#include <viskores/cont/ArrayHandleRuntimeVec.h>
#include <viskores/cont/ArrayCopy.h>


// Uncomment to use the closed-form version of the inverse linear interpolation
// (ilerp) uncertainty function. Comment to use Monte Carlo sampling instead.
#define KDE_MAGIC_USE_CLOSED_FORM_ILERP

namespace
{
/* Worklet to compute the most probable isosurface topology.
"inputVectors" (input): Ensemble data,
"outputMagnitudes" (output): Scalar field representing the most probable isosurface topology (or grid vertex signs),
"isovalueData" (input): Isovalue.
 */
struct ComputeMostProbableTopologyNonOptimized : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn inputVectors, FieldOut outputMagnitudes, WholeArrayIn isovalueData);
    
  // ExecutionSignature maps to the operator's arguments
  using ExecutionSignature = void(_1, _2, _3);

  template <typename T, typename T2, typename T3>
  VISKORES_EXEC void operator()(const T& inVector, T2& outMagnitude, const T3& isovalue) const
    {
        // Isovalue
        viskores::Float64 isoVal = isovalue.Get(0);
        
        // Number of ensemble members
        viskores::IdComponent numComponents = inVector.GetNumberOfComponents();
        
        //Compute bandwidth using the Silverman's rule of thumb
        float sampleMean = 0;
        float sampleVar = 0;
        float* samples = new float[numComponents];
        
        // Quant1 and quant 2 are specific to Gaussian kernel
        viskores::FloatDefault quant1 = 1.0;
        viskores::FloatDefault pi = viskores::Pi();
        viskores::FloatDefault quant2 = (1.0/(2.0*viskores::Sqrt(pi)));
        std::vector<viskores::FloatDefault> myVector;
          
        // Compute mean of data
        for(int i=0; i<numComponents; i++)
        {
            sampleMean+=inVector[i];
            // Copy to arrays of type double
            samples[i] = (float)inVector[i];
            myVector.push_back(inVector[i]);
        }
        sampleMean = sampleMean/(double)numComponents;
             
        // Compute data variance
        for(int i=0; i<numComponents; i++)
        {
            sampleVar+= pow((double)inVector[i] - sampleMean, 2);
        }

        sampleVar = sampleVar/(double)numComponents;
    
        // Standard deviation of samples
        viskores::FloatDefault sampleDev = viskores::Sqrt(sampleVar);
        
        // Calculate IQR of data. Make sure that number of samples/ensemble members is at least 4
        std::sort(myVector.begin(), myVector.end());
        size_t n = myVector.size();
        
        // IQR: Calculate median of lower half
        std::vector<double> lower_half(myVector.begin(), myVector.begin() + n / 2);
        viskores::FloatDefault low_median;
        size_t lower_n = lower_half.size();
        if (lower_n % 2 == 1) {
            low_median =  lower_half[lower_n / 2];
        } else {
            low_median = (lower_half[lower_n / 2 - 1] + lower_half[lower_n / 2]) / 2.0;
        }
        
        // IQR: Calculate median of upper half
        std::vector<double> upper_half;
        if (n % 2 == 1) {
            upper_half.assign(myVector.begin() + n / 2 + 1, myVector.end());
        } else {
                upper_half.assign(myVector.begin() + n / 2, myVector.end());
        }
        viskores::FloatDefault up_median;
        size_t upper_n = upper_half.size();
        if (upper_n % 2 == 1) {
            up_median =  upper_half[upper_n / 2];
        } else {
            up_median = (upper_half[upper_n / 2 - 1] + upper_half[upper_n / 2]) / 2.0;
        }
        
        // IQR: Difference between upper and lower half median
        viskores::FloatDefault iqr = up_median - low_median;
        viskores::FloatDefault scaled_iqr = iqr/1.34;
        
        // Spread is minimum of IQR and standard deviation
        viskores::FloatDefault spread = viskores::Min(scaled_iqr,sampleDev);
        
        // Kernel  bandwidth with Silverman's rule of thumb for a Gaussian kernel
        viskores::FloatDefault coeff1 = pow(quant1,-2.0/5.0)* pow(quant2, 1.0/5.0);
        viskores::FloatDefault coeff2;
        
        if (viskores::Abs(spread) < 0.0001)
        {
            spread = 0.0001;
            coeff2 =  pow(((3.0/8.0)* pow(pi,(-1.0/2.0))* pow(spread,-5.0)),(-1.0/5.0));
        }
        else
        {
            coeff2 =  pow(((3.0/8.0)* pow(pi,(-1.0/2.0))* pow(spread,-5.0)),(-1.0/5.0));
        }
        viskores::FloatDefault coeff3 = pow(numComponents,-1.0/5.0);
        viskores::FloatDefault kernel_bandwidth = coeff1*coeff2*coeff3;
        
        // Derive most probable vertex sign and create relevant data grid
        viskores::FloatDefault probNegativeSign = 0;
        viskores::FloatDefault probPositiveSign = 0;
        viskores::Id numComponentsGreaterThanIsovalue = 0;
        viskores::FloatDefault sumOfComponentsGreaterThanIsovalue = 0;
        viskores::FloatDefault sumOfComponentsSmallerThanIsovalue = 0;
        
        // Compute the probability of vertex sign to be negative
        for (int i = 0; i < numComponents; i++)
        {
            viskores::Float64 kernelCenter = inVector[i];
            
            probNegativeSign = probNegativeSign + 0.5*(1 + std::erf((isoVal - kernelCenter)/(viskores::Sqrt(2)*kernel_bandwidth)));
            
            if (kernelCenter >= isoVal)
            {
                numComponentsGreaterThanIsovalue = numComponentsGreaterThanIsovalue+1;
                sumOfComponentsGreaterThanIsovalue = sumOfComponentsGreaterThanIsovalue + kernelCenter;
            }
            else
            {
                sumOfComponentsSmallerThanIsovalue = sumOfComponentsSmallerThanIsovalue + kernelCenter;
            }
            
        }
        
        probNegativeSign = (double) probNegativeSign/numComponents;
        probPositiveSign = 1.0 - probNegativeSign;
        
        if (probPositiveSign >= 0.5)
            // Assign the mean of values greater than the isovalue, if most probable vertex sign is positive
            outMagnitude = sumOfComponentsGreaterThanIsovalue/numComponentsGreaterThanIsovalue;
        else
            // Assign the mean of values smaller than the isovalue, if most probable vertex sign is negative
            outMagnitude = sumOfComponentsSmallerThanIsovalue/(numComponents - numComponentsGreaterThanIsovalue);
    }
};

/* Worklet to compute the mean-field isosurface topology.
"inputVectors" (input): Ensemble data,
"outputMagnitudes" (output): Scalar field representing mean of the ensemble data.
 */
struct ComputeMean : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn inputVectors, FieldOut outputMagnitudes);

  // InPointFieldType should be a vector
  //template <typename InPointFieldVecEnsemble,
  //          typename OutPointFieldType1,>
  template <typename T, typename T2>
  VISKORES_EXEC void operator()(const T& inVector, T2& outMagnitude) const
  {
      // Number of ensemble members
      viskores::IdComponent numComponents = inVector.GetNumberOfComponents();
      viskores::FloatDefault meanValue = 0;
      for (int i = 0; i < numComponents; i++)
      {
          meanValue+=inVector[i];
      }
      meanValue = meanValue/numComponents;
      outMagnitude = meanValue;
  }
};


/* Worklet to compute isosurface crossing uncertainty on a grid edge (KDE MAGIC worklet).
"edgeIds": Input Ids of edges estimated to be crossed by the isosurface under uncertainty,
"interpMean" (input): Isovalue,
"inputMeans" (input): Most probable/Mean field depending on the worklet used to compute probable vertex signs,
"inputEnsemble" (input): Variance field (variance/uncertainty in the mean values),
"outputVariance" (output): Variance of isosurface crossing on a grid edge,
"outputExpectedCrossing" (output): Expected crossing position of isosurface on a grid edge.
 */
struct ComputeEdgeVarianceWorklet : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn edgeIds,
                                FieldIn interpMean,
                                WholeArrayIn inputMeans,
                                WholeArrayIn inputEnsemble,
                                FieldOut outputVariance,
                                FieldOut expectedCrossing);
  //using ExecutionSignature = _5(_1, _2, _3, _4);

  template <typename T, typename MeanPortalType, typename ensemblePortalType>
  VISKORES_EXEC void operator()(const viskores::Id2& edgeIds,
                             T interpMean,
                             const MeanPortalType& inputMeans,
                             const ensemblePortalType& inputEnsembles,
                             viskores::Float64& outputEdgeVariance,
                             viskores::Float64& outputExpectedCrossing) const
  {
    // Extract ensemble at each edge vertex
    auto ensemble0 = inputEnsembles.Get(edgeIds[0]);
    auto ensemble1 = inputEnsembles.Get(edgeIds[1]);
      
    // Number of ensemble members/kernels
    viskores::IdComponent numKernels = ensemble0.GetNumberOfComponents();
   
    // Call a function to compute the inverse linear interpolation uncertainty
    gaussian_distribution g;
    g.setNumGaussiansInKde1(numKernels);
    g.setNumGaussiansInKde2(numKernels);
      
    float s0Mean = 0;
    float s1Mean = 0;
    float s0Var = 0;
    float s1Var = 0;

    double coVar;
      
    float* ens0 = new float[numKernels];
    float* ens1 = new float[numKernels];
      
    // Compute the mean of the ensemble data
    for(int i=0; i<numKernels; i++)
    {
        s0Mean+=ensemble0[i];
        s1Mean+=ensemble1[i];
        
        // Copy to arrays of type double
        ens0[i] = (float)ensemble0[i];
        ens1[i] = (float)ensemble1[i];
        
    }
    s0Mean = s0Mean/(double)numKernels;
    s1Mean = s1Mean/(double)numKernels;
         
    // Compute the variance of the ensemble data
    for(int i=0; i<numKernels; i++)
    {
        s0Var+= pow((double)ensemble0[i] - s0Mean, 2);
        s1Var+= pow((double)ensemble1[i] - s1Mean, 2);
    }

    s0Var = s0Var/(double)numKernels;
    s1Var = s1Var/(double)numKernels;

    // Kernel bandwidth for a bivariate kernel
    float sf = pow(numKernels*(2.0 + 2.0) / 4.0, -1. / (2.0 + 4.0));
    float dev1 = s0Var*sf*sf;
    float dev2 = s1Var*sf*sf;

    double expectedConsiderCorr, c_probConsiderCorr, sigConsiderCorr, varConsiderCorr;
    
#ifdef KDE_MAGIC_USE_CLOSED_FORM_ILERP
    // Closed-form computation.
    int option = 1;
      
    // Set the dummy sample count
    int numSamples = 100;

    g.kde_gaussian_alpha_pdf(ens0,
                             &dev1,
                             ens1,
                             &dev2,
                             0.0,
                             interpMean,
                             &expectedConsiderCorr,
                             &c_probConsiderCorr,
                             &varConsiderCorr,
                             numSamples,
                             option);
#else
    // Monte Carlo computation
    int option = 0;
        
    // Set the sample count
    int numSamples = 100;

    g.kde_gaussian_alpha_pdf(ens0,
                             &dev1,
                             ens1,
                             &dev2,
                             0.0,
                             interpMean,
                             &expectedConsiderCorr,
                             &c_probConsiderCorr,
                             &varConsiderCorr,
                             numSamples,
                             option);
#endif
    
    outputExpectedCrossing = expectedConsiderCorr;
    outputEdgeVariance = varConsiderCorr;
      
  }
};

/* Function to call isosurface uncertainty worklet for a grid edge.
"ensembleArray" (input): Array handle representing the ensemble data,
"inputMeanArrayUnknown" (input): Array handle representing  the mean/most probable field,
"outputMeanArrayUnknown" (input): Array handle representing same isovalue for a worklet,
"edgeIdsUnknown" (input): Array handle of Ids of edges estimated to be crossed by the isosurface under uncertainty,
"outputEdgeVariance" (output): Array handle represnting variance of isosurface crossing on a grid edge,
"expectedCrossing" (output): Array handle representing expected crossing position of isosurface on a grid edge.
 */
template <typename EnsembleArrayType, typename EdgeVarianceArrayType>
VISKORES_CONT void ComputeEdgeVariance(
  const EnsembleArrayType& ensembleArray,
  const viskores::cont::UnknownArrayHandle& inputMeanArrayUnknown,
  const viskores::cont::UnknownArrayHandle& outputMeanArrayUnknown,
  const viskores::cont::UnknownArrayHandle& edgeIdsUnknown,
  EdgeVarianceArrayType& outputEdgeVariance,
  viskores::cont::ArrayHandle<viskores::Float64>& expectedCrossing
)
{
    // use std::decay to remove const ref from the decltype of concrete.
    using T = typename EnsembleArrayType::ValueType::ComponentType;
    using ReturnType = typename ::viskores::detail::FloatingPointReturnType<T>::Type;
    viskores::cont::ArrayHandle<ReturnType> inputMeanArray;
    inputMeanArrayUnknown.AsArrayHandle(inputMeanArray);
    viskores::cont::ArrayHandle<ReturnType> outputMeanArray;
    outputMeanArrayUnknown.AsArrayHandle(outputMeanArray);

  viskores::cont::ArrayHandle<viskores::Id2> edgeIds;
  edgeIdsUnknown.AsArrayHandle(edgeIds);

  viskores::cont::Invoker invoke;
  invoke(ComputeEdgeVarianceWorklet{},
         edgeIds,
         outputMeanArray,
         inputMeanArray,
         ensembleArray,
         outputEdgeVariance,
         expectedCrossing);
}

/* Worklet to incorporate expected crossing positions (computed with the KDE MAGIC worklet) for isosurface visualization.
 "edgeIds" (input): Ids of edges estimated to be crossed by the isosurface under uncertainty,
 "crossings" (input): Expected crossing position of isosurface on a grid edge computed using MAGIC [0,1],
 "inputArray" (input): Mean field,
 "interpolated" (output): Isourface vertex physical position based on expected crossing.
*/
struct InterpolateFieldWorklet : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn edgeIds,
                                FieldIn crossings,
                                WholeArrayIn inputArray,
                                FieldOut interpolated);

  template <typename InputPortalType, typename InterpolatedType>
  VISKORES_EXEC void operator()(const viskores::Id2& edgeIds,
                                viskores::Float64 crossing,
                                const InputPortalType& inputArray,
                                InterpolatedType& interpolated) const
  {
    using VecTOut = viskores::VecTraits<InterpolatedType>;
    const viskores::IdComponent numComponents = VecTOut::GetNumberOfComponents(interpolated);

    auto input0 = inputArray.Get(edgeIds[0]);
    auto input1 = inputArray.Get(edgeIds[1]);

    using VecTIn = viskores::VecTraits<decltype(input0)>;
    VISKORES_ASSERT(VecTIn::GetNumberOfComponents(input0) == numComponents);
    VISKORES_ASSERT(VecTIn::GetNumberOfComponents(input1) == numComponents);

    for (viskores::IdComponent componentIdx = 0; componentIdx < numComponents; ++componentIdx)
    {
      auto v0 = VecTIn::GetComponent(input0, componentIdx);
      auto v1 = VecTIn::GetComponent(input1, componentIdx);
      VecTOut::SetComponent(interpolated, componentIdx, viskores::Lerp(v0, v1, crossing));
    }
  }
};

/* Function to call a worklet that uses expected crossing location (computed with the KDE MAGIC worklet) for isosurface visualization.
"inputArray" (input): Array handle of mean field values,
"edgeIdsUnknown" (input): Array handle of Ids of edges estimated to be crossed by the isosurface under uncertainty,
"crossings" (input): Array handle of expected crossing position of isosurface on a grid edge computed using MAGIC [0,1],
"outputArray" (output): Array handle of isourface vertex physical position based on KDE MAGIC expected crossing.
*/
template <typename InputArrayType>
VISKORES_CONT viskores::cont::UnknownArrayHandle InterpolateField(
  const InputArrayType& inputArray,
  const viskores::cont::UnknownArrayHandle& edgeIdsUnknown,
  const viskores::cont::ArrayHandle<viskores::Float64>& crossings)
{
  using ValueType = typename InputArrayType::ValueType;
  using ComponentType = typename viskores::VecTraits<ValueType>::BaseComponentType;

  viskores::cont::ArrayHandleRuntimeVec<ComponentType> outputArray(
    inputArray.GetNumberOfComponentsFlat());

  viskores::cont::ArrayHandle<viskores::Id2> edgeIds;
  edgeIdsUnknown.AsArrayHandle(edgeIds);
  VISKORES_ASSERT(edgeIds.GetNumberOfValues() == crossings.GetNumberOfValues());

  viskores::cont::Invoker invoke;
  invoke(InterpolateFieldWorklet{}, edgeIds, crossings, inputArray, outputArray);

  return outputArray;
}

} // anonymous namespace

/* Filter for isosurface uncertainty computation using the KDE MAGIC worklet.
"inDataSet" (input): Input ensemble dataset,
"outputData" (output): Output data of isourface expected vertex positions and their spatial variance.
*/
viskores::cont::DataSet KDE_MAGIC::DoExecute(const viskores::cont::DataSet& inDataSet)
{
    // Compute mean of the ensemble
    /*viskores::cont::UnknownArrayHandle meanArray;
    auto resolveType = [&](const auto& concrete)
    {
       // use std::decay to remove const ref from the decltype of concrete.
       using T = typename std::decay_t<decltype(concrete)>::ValueType::ComponentType;
       using ReturnType = typename ::viskores::detail::FloatingPointReturnType<T>::Type;
       viskores::cont::ArrayHandle<ReturnType> result;

       this->Invoke(ComputeMean{}, concrete, result);
       meanArray = result;
    };
    const auto& field = this->GetFieldFromDataSet(inDataSet);
    field.GetData().CastAndCallWithExtractedArray(resolveType);*/
    
    // Compute most probable topology for the ensemble
    viskores::cont::UnknownArrayHandle meanArray;
    auto resolveType = [&](const auto& concrete)
    {
        // use std::decay to remove const ref from the decltype of concrete.
        using T = typename std::decay_t<decltype(concrete)>::ValueType::ComponentType;
        using ReturnType = typename ::viskores::detail::FloatingPointReturnType<T>::Type;
        viskores::cont::ArrayHandle<ReturnType> result;
        viskores::Float64 isovalue = this->IsoValues[0];
        
        // Create a one-element ArrayHandle for the constant number (here, isovalue)
        viskores::cont::ArrayHandle<viskores::Float64> constantArray = viskores::cont::make_ArrayHandle({isovalue});
        this->Invoke(ComputeMostProbableTopologyNonOptimized{}, concrete, result, constantArray);
        meanArray = result;
    };
    const auto& field = this->GetFieldFromDataSet(inDataSet);
    field.GetData().CastAndCallWithExtractedArray(resolveType);
    
    // Name the most probable field
    std::string meanFieldName = field.GetName() + "_mean";
    viskores::cont::DataSet meanData = inDataSet;
    meanData.AddField(meanFieldName, field.GetAssociation(), meanArray);
    
    // Do the actual contour extract while saving the edge information.
    viskores::filter::contour::Contour contourExtract;
    contourExtract.SetIsoValues(this->IsoValues);
    contourExtract.SetGenerateNormals(this->GetGenerateNormals());
    contourExtract.SetComputeFastNormals(this->GetComputeFastNormals());
    contourExtract.SetNormalArrayName(this->GetNormalArrayName());
    contourExtract.SetMergeDuplicatePoints(this->GetMergeDuplicatePoints());
    contourExtract.SetActiveField(0, meanFieldName, field.GetAssociation());

    // Select fields to pass to the contour filter.
    viskores::filter::FieldSelection fieldSelection = this->GetFieldsToPass();
    for (viskores::IdComponent fieldId = 0; fieldId < meanData.GetNumberOfFields(); ++fieldId)
    {
        viskores::cont::Field field = meanData.GetField(fieldId);
        if (field.IsPointField())
        {
          // Do not interpolate point fields as they will have to be reinterpolated later.
          fieldSelection.AddField(field, viskores::filter::FieldSelection::Mode::Exclude);
        }
    }
    
    // Use the most probable grid (computed above) to extract the isosurface topology.
    fieldSelection.AddField(this->GetActiveFieldName(0),
                            this->GetActiveFieldAssociation(0),
                            viskores::filter::FieldSelection::Mode::Select);

    // Save edge information representing IDs of edges crossed by the isosurface
    contourExtract.SetAddInterpolationEdgeIds(true);
    
    // Extract the isosurface
    viskores::cont::DataSet contours = contourExtract.Execute(meanData);
    
    // Pass array handles representing the ensemble data, most probable field, isovalue, edge ID (estimaged to be crossed by the isosurface) to compute isosurface expected position and variance on grid edgs using the KDE MAGIC worklet.
    viskores::cont::UnknownArrayHandle outputVariance;
    viskores::cont::ArrayHandle<viskores::Float64> expectedCrossings;
    auto resolveType2 = [&](const auto& concrete)
    {
        viskores::cont::ArrayHandle<viskores::Float64> variance;

        // Call the KDE MAGIC worklet
        ComputeEdgeVariance(concrete,
                            meanArray,
                            contours.GetField(meanFieldName).GetData(),
                            contours.GetField("edgeIds").GetData(),
                            variance,
                            expectedCrossings);
        outputVariance = variance;
    };
    field.GetData().CastAndCallWithExtractedArray(resolveType2);
    
    // Use expected isosurface crossing locations on grid edges computed with KDE MAGIC ([0,1]) to compute physical isosurface vertex locations
    auto interpField = [&](const auto& inputArray, viskores::cont::UnknownArrayHandle& outputArray)
     {
       outputArray =
         InterpolateField(inputArray, contours.GetField("edgeIds").GetData(), expectedCrossings);
     };

    auto mapField = [&](viskores::cont::DataSet& outputData, const viskores::cont::Field& inputField){
        if (inputField.IsPointField())
        {
            viskores::cont::UnknownArrayHandle outputArray;
            this->CastAndCallVariableVecField(inputField.GetData(), interpField, outputArray);
            outputData.AddPointField(inputField.GetName(), outputArray);
        }
        else
        {
            outputData.AddField(inputField); // pass through
        }
    };
    viskores::cont::DataSet outputData = this->CreateResult(meanData, contours.GetCellSet(), mapField);

    // Add the variance of isosurface vertex positions to the output dataset (computed with the KDE MAGIC worklet).
    outputData.AddPointField("variance_edge_crossing", outputVariance);

    // Add the expected positions of isosurface vertex positions to the output dataset (computed with the KDE MAGIC worklet).
    outputData.AddPointField("expected_edge_crossing", expectedCrossings);

    return outputData;
}
