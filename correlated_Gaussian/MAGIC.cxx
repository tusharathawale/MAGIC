/*============================================================================

Title   -  MAGIC: Marching Cubes Isosurface Uncertainty Visualization for Gaussian Uncertain Data with Spatial Correlation
Authors -  Tushar M. Athawale, Kenneth Moreland, David Pugmire, Chris R. Johnson, Paul Rosen, Matthew Norman, Antigoni Georgiadou, and Alireza Entezari
Date    -  Oct 20, 2025

This code is distributed under the OSI-approved BSD 3-clause License. See LICENSE.txt for details.
 
============================================================================*/

#include "MAGIC.h"

#include <viskores/cont/ArrayCopy.h>
#include <viskores/cont/CellSetStructured.h>
#include <viskores/cont/Algorithm.h>
#include <viskores/Math.h>
#include <viskores/worklet/WorkletMapField.h>

// Uncomment to use the closed-form version of the inverse linear interpolation
// (ilerp) uncertainty function. Comment to use Monte Carlo sampling instead.
#define MAGIC_USE_CLOSED_FORM_ILERP

namespace
{

/* Worklet to compute isosurface crossing uncertainty on a grid edge (MAGIC worklet).
"edgeIds": Input Ids of edges estimated to be crossed by the isosurface under uncertainty,
"interpMean" (input): Isovalue,
"inputMeans" (input): Mean field,
"inputVariance" (input): Variance field (variance/uncertainty in the mean values),
"inputRhoX" (input): Covariance field along the X physical dimension,
"inputRhoX" (input): Covariance field along the Y physical dimension,
"inputRhoZ" (input): Covariance field along the Z physical dimension,
"outputEdgeVariance" (output): Variance of isosurface crossing on a grid edge,
"outputExpectedCrossing" (output): Expected crossing position of isosurface on a grid edge.
 */
struct ComputeEdgeVarianceWorklet : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn edgeIds,
                                FieldIn interpMean,
                                WholeArrayIn inputMeans,
                                WholeArrayIn inputVariance,
                                WholeArrayIn inputRhoX,
                                WholeArrayIn inputRhoY,
                                WholeArrayIn inputRhoZ,
                                FieldOut outputVariance,
                                FieldOut expectedCrossing);

  VISKORES_CONT ComputeEdgeVarianceWorklet(const viskores::Id3 resolution)
    : Resolution(resolution)
  {}

  template <typename T, typename PortalType>
  VISKORES_EXEC void operator()(const viskores::Id2& edgeIds,
                                T interpMean,
                                const PortalType& inputMeans,
                                const PortalType& inputVariance,
                                const PortalType& inputRhoX,
                                const PortalType& inputRhoY,
                                const PortalType& inputRhoZ,
                                T& outputEdgeVariance,
                                viskores::Float64& outputExpectedCrossing) const
  {
    //Extract mean and variance at each edge vertex
    T mean0 = inputMeans.Get(edgeIds[0]);
    T mean1 = inputMeans.Get(edgeIds[1]);
    T variance0 = inputVariance.Get(edgeIds[0]);
    T variance1 = inputVariance.Get(edgeIds[1]);
        
    // Convert edge end vertex positions (respresented) to physical X, Y, and Z coordinates
    // Might need to adjust the code below for other datasets depending on which dimension
    // is referred to 0 (i.e., X) and 1 (i.e., Y) in other datasets
    viskores::Id a0 = edgeIds[0]/(this->Resolution[1]*this->Resolution[0]);
    viskores::Id b0 = (edgeIds[0]%(this->Resolution[1]*this->Resolution[0]))/this->Resolution[0];
    viskores::Id c0 = edgeIds[0]%this->Resolution[0];
            
    viskores::Id a1 = edgeIds[1]/(this->Resolution[1]*this->Resolution[0]);
    viskores::Id b1 = (edgeIds[1]%(this->Resolution[1]*this->Resolution[0]))/this->Resolution[0];
    viskores::Id c1 = edgeIds[1]%this->Resolution[0];

    viskores::Float64 covar=0.0;
      
    // Extract covariance for data at edge ends depending on if the edge is going in the X, Y, or Z direction
    if (viskores::Abs(a1 - a0) == 1)
    {
        if(a0 < a1)
            covar = inputRhoX.Get(edgeIds[0]);
        else
            covar = inputRhoX.Get(edgeIds[1]);
    }
        
    else if (viskores::Abs(b1 - b0) == 1)
    {
        if(b0 < b1)
            covar = inputRhoY.Get(edgeIds[0]);
        else
            covar = inputRhoY.Get(edgeIds[1]);
    }
        
    else if (viskores::Abs(c1 - c0) == 1)
    {
        if(c0 < c1)
            covar = inputRhoZ.Get(edgeIds[0]);
        else
            covar = inputRhoZ.Get(edgeIds[1]);
    }
      
   // Call a function to compute the inverse linear interpolation uncertainty
    gaussian_distribution g;
    viskores::Float64 cross_prob;
    viskores::Float64 var;
    
#ifdef MAGIC_USE_CLOSED_FORM_ILERP
    // Closed-form computation
    g.gaussian_alpha_pdf(mean0,
                         variance0,
                         mean1,
                         variance1,
                         covar,
                         interpMean,
                         &outputExpectedCrossing,
                         &cross_prob,
                         &var);
    
#else
    // Monte Carlo computation (for nonsingular covariance matrix that commonly occurs in the real data)
    // The Monte Carlo function - version v2 - below works ok with OpenMP
    
    // Number of Monte Carlo samples
    viskores::Id numSamples = 4000;
    /*
    g.gaussian_alpha_pdf_MonteCarlo_v2(mean0,
                                       variance0,
                                       mean1,
                                       variance1,
                                       covar,
                                       interpMean,
                                       &outputExpectedCrossing,
                                       &cross_prob,
                                       &var,
                                       numSamples);*/
    
    // Use the following Monte Carlo computation for the "perfectly correlated" (correlation=1) data.
    // The perfectly correlated data can result in a singular covariance matrix (with determinant 0) and therefore becoming noninvertible with inability
    // to generate samples. For the tangle dataset with perfect correlation in the TVCG paper, the covariance value should
    // be slightly reduced (with correlation reduced from 1 to 0.99) and set as below to successfully generate
    // samples and avoid singular covariance matrix!!!!
    // The Monte Carlo function - version v2 - below works ok with OpenMP
    
    viskores::Float64 covar_perefect_correlation=0.99*viskores::Sqrt(variance0)*viskores::Sqrt(variance1);
    g.gaussian_alpha_pdf_MonteCarlo_v2(mean0,
                                       variance0,
                                       mean1,
                                       variance1,
                                       covar_perefect_correlation,
                                       interpMean,
                                       &outputExpectedCrossing,
                                       &cross_prob,
                                       &var,
                                       numSamples);
    
#endif
    //Uncomment below for the computation of difference between the Monte Carlo and closed-form solutions.
    //You will also need to use the Monte Carlo code above.
    
    /*viskores::Float64 varTemp;
    g.gaussian_alpha_pdf(mean0,
                         variance0,
                         mean1,
                         variance1,
                         covar,
                         interpMean,
                         &outputExpectedCrossing,
                         &cross_prob,
                         &varTemp);
    var = viskores::Abs(var-varTemp);*/
      
    outputEdgeVariance = var;
  }

private:
  viskores::Id3 Resolution;
};


/* Function to call isosurface uncertainty worklet for a grid edge.
"variance" (input): Array handle representing the pointwise variance field (variance/uncertainty in the mean values),
"inputRhoXArrayUnknown" (input): Array handle representing the covariance field in X direction on grid edges,
"inputRhoYArrayUnknown" (input): Array handle representing the covariance field in Y direction on grid edges,
"inputRhoZArrayUnknown" (input): Array handle representing the covariance field in Z direction on grid edges,
"outputMeanArrayUnknown" (input): Array handle representing same isovalue for a worklet,
"inputMeanArrayUnknown" (input): Array handle representing  the mean field,
"edgeIdsUnknown" (input): Array handle of Ids of edges estimated to be crossed by the isosurface under uncertainty,
"outputEdgeVariance" (output): Array handle represnting variance of isosurface crossing on a grid edge,
"expectedCrossing" (output): Array handle representing expected crossing position of isosurface on a grid edge.
 */
template <typename VarianceArrayType, typename EdgeVarianceArrayType>
VISKORES_CONT void ComputeEdgeVariance(
  const VarianceArrayType& variance,
  const viskores::cont::UnknownArrayHandle& inputRhoXArrayUnknown,
  const viskores::cont::UnknownArrayHandle& inputRhoYArrayUnknown,
  const viskores::cont::UnknownArrayHandle& inputRhoZArrayUnknown,
  const viskores::cont::UnknownArrayHandle& outputMeanArrayUnknown,
  const viskores::cont::UnknownArrayHandle& inputMeanArrayUnknown,
  const viskores::cont::UnknownArrayHandle& edgeIdsUnknown,
  viskores::Id3 resolution, EdgeVarianceArrayType& outputEdgeVariance,
  viskores::cont::ArrayHandle<viskores::Float64>& expectedCrossing
)
{
  VarianceArrayType outputMeanArray;
  viskores::cont::ArrayCopyShallowIfPossible(outputMeanArrayUnknown, outputMeanArray);
    
  VarianceArrayType inputMeanArray;
  viskores::cont::ArrayCopyShallowIfPossible(inputMeanArrayUnknown, inputMeanArray);
    
  VarianceArrayType inputRhoXArray;
  viskores::cont::ArrayCopyShallowIfPossible(inputRhoXArrayUnknown, inputRhoXArray);
  
  VarianceArrayType inputRhoYArray;
  viskores::cont::ArrayCopyShallowIfPossible(inputRhoYArrayUnknown, inputRhoYArray);
    
  VarianceArrayType inputRhoZArray;
  viskores::cont::ArrayCopyShallowIfPossible(inputRhoZArrayUnknown, inputRhoZArray);
  
  viskores::cont::ArrayHandle<viskores::Id2> edgeIds;
  edgeIdsUnknown.AsArrayHandle(edgeIds);

  viskores::cont::Invoker invoke;
  invoke(ComputeEdgeVarianceWorklet{ resolution },
         edgeIds,
         outputMeanArray,
         inputMeanArray,
         variance,
         inputRhoXArray,
         inputRhoYArray,
         inputRhoZArray,
         outputEdgeVariance,
         expectedCrossing);
    
  // Quantitative evaluation: For calculating aggregate diff between Monte Carlo and closed
  /*viskores::FloatDefault sum = viskores::cont::Algorithm::Reduce(outputEdgeVariance, 0.0f, viskores::Add());
  std::cout<<"Difference in Monte Carlo and Closed-form variance:"<<sum<<"\n";*/
}

/* Worklet to incorporate expected crossing positions (computed with the MAGIC worklet) for isosurface visualization.
 "edgeIds" (input): Ids of edges estimated to be crossed by the isosurface under uncertainty,
 "crossings" (input): Expected crossing position of isosurface on a grid edge computed using MAGIC [0,1],
 "inputArray" (input): Mean field,
 "interpolated" (output): isourface vertex physical position based on expected crossing (computed with the MAGIC worklet).
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

/* Function to call a worklet that uses expected crossing location (computed with the MAGIC worklet) for isosurface visualization.
"inputArray" (input): Array handle of mean field values,
"edgeIdsUnknown" (input): Array handle of Ids of edges estimated to be crossed by the isosurface under uncertainty,
"crossings" (input): Array handle of expected crossing position of isosurface on a grid edge computed using MAGIC [0,1],
"outputArray" (output): Array handle of isourface vertex physical position based on MAGIC expected crossing.
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

/* Filter for isosurface uncertainty computation using the MAGIC worklet.
"input" (input): Input data with five fields, namely, the mean, pointwise variance, covariance in X, covariance in Y, and covariance in Z
"outputData" (output): Output data of isourface expected vertex positions and their spatial variance.
*/
viskores::cont::DataSet MAGIC::DoExecute(const viskores::cont::DataSet& input)
{
  // Get the resolution of the input (which must be structured) so that we can
  // find the edge data.
  viskores::cont::CellSetStructured<3> cellSet;
  input.GetCellSet().AsCellSet(cellSet);
  viskores::Id3 resolution = cellSet.GetPointDimensions();

  // Do the actual contour extract while saving the edge information.
  viskores::filter::contour::Contour contourExtract;
  contourExtract.SetIsoValues(this->IsoValues);
  contourExtract.SetGenerateNormals(this->GetGenerateNormals());
  contourExtract.SetComputeFastNormals(this->GetComputeFastNormals());
  contourExtract.SetNormalArrayName(this->GetNormalArrayName());
  contourExtract.SetMergeDuplicatePoints(this->GetMergeDuplicatePoints());
  contourExtract.SetActiveField(0, this->GetActiveFieldName(0), this->GetActiveFieldAssociation());

  viskores::filter::FieldSelection fieldSelection = this->GetFieldsToPass();
    
  //Select fields to pass to the contour filter.
  for (viskores::IdComponent fieldId = 0; fieldId < input.GetNumberOfFields(); ++fieldId)
  {
     viskores::cont::Field field = input.GetField(fieldId);
     if (field.IsPointField())
     {
        // Do not interpolate point fields as they will have to be reinterpolated later.
        fieldSelection.AddField(field, viskores::filter::FieldSelection::Mode::Exclude);
     }
  }

    
  // Use the mean information to extract the isosurface topology (since it is most probable for the Gaussian model).
  fieldSelection.AddField(this->GetActiveFieldName(0),
                          this->GetActiveFieldAssociation(0),
                          viskores::filter::FieldSelection::Mode::Select);
  contourExtract.SetFieldsToPass(fieldSelection);

  // Save edge information representing IDs of edges crossed by the isosurface
  contourExtract.SetAddInterpolationEdgeIds(true);

  // Extract the isosurface
  viskores::cont::DataSet contours = contourExtract.Execute(input);

  // Pass array handeles representing the pointwise variance field, covariance in X, covariance in Y, covariance in Z,
  // isovalue, mean field, edge ID (crossed by the isosurface) to compute isosurface expected position and variance
  // on grid edgs using MAGIC.
  viskores::cont::UnknownArrayHandle outputVariance;
  viskores::cont::ArrayHandle<viskores::Float64> expectedCrossings;
  auto resolveArray = [&](auto inputVarianceArray)
  {
    // pass variance array, rhoX, rhoY, and rhoZ, mean from contour, mean form input, edge Ids
    using T = typename std::decay_t<decltype(inputVarianceArray)>::ValueType;
    viskores::cont::ArrayHandle<T> variance;
    ComputeEdgeVariance(inputVarianceArray,
                                         this->GetFieldFromDataSet(2, input).GetData(),
                                         this->GetFieldFromDataSet(3, input).GetData(),
                                         this->GetFieldFromDataSet(4, input).GetData(),
                                         this->GetFieldFromDataSet(0, contours).GetData(),
                                         this->GetFieldFromDataSet(0, input).GetData(),
                                         contours.GetField("edgeIds").GetData(),
                                         resolution, variance, expectedCrossings);
      
    outputVariance = variance;
     
  };
  this->CastAndCallScalarField(this->GetFieldFromDataSet(1, input), resolveArray);

  // Use expected isosurface crossing locations on grid edges computed with MAGIC ([0,1]) to compute physical isosurface vertex locations
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
   viskores::cont::DataSet outputData = this->CreateResult(input, contours.GetCellSet(), mapField);

   // Add the variance of isosurface vertex positions to the output dataset (computed with MAGIC).
   outputData.AddPointField("variance_edge_crossing", outputVariance);
    
   // Add the expected positions of isosurface vertex positions to the output dataset (computed with MAGIC).
   outputData.AddPointField("expected_edge_crossing", expectedCrossings);

   return outputData;
}
