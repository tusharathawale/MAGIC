/*============================================================================

Title   -  MAGIC: Marching Cubes Isosurface Uncertainty Visualization for Gaussian Uncertain Data with Spatial Correlation
Authors -  Tushar M. Athawale, Kenneth Moreland, David Pugmire, Chris R. Johnson, Paul Rosen, Matthew Norman, Antigoni Georgiadou, and Alireza Entezari
Date    -  Oct 20, 2025

This code is distributed under the OSI-approved BSD 3-clause License. See LICENSE.txt for details.
 
============================================================================*/

#include "MAGIC.h"
#include "gaussian_distribution.h"

#include <viskores/cont/ArrayCopy.h>
#include <viskores/cont/Algorithm.h>
#include <viskores/cont/ArrayHandleRuntimeVec.h>
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
"outputVariance" (output): Variance of isosurface crossing on a grid edge,
"outputExpectedCrossing" (output): Expected crossing position of isosurface on a grid edge.
 */
struct ComputeEdgeVarianceWorklet : viskores::worklet::WorkletMapField
{
  using ControlSignature = void(FieldIn edgeIds,
                                FieldIn interpMean,
                                WholeArrayIn inputMeans,
                                WholeArrayIn inputVariance,
                                FieldOut outputVariance,
                                FieldOut expectedCrossing);

  template <typename T, typename PortalType>
  VISKORES_EXEC void operator()(const viskores::Id2& edgeIds,
                                T interpMean,
                                const PortalType& inputMeans,
                                const PortalType& inputVariance,
                                T& outputEdgeVariance,
                                viskores::Float64& outputExpectedCrossing) const
  {
    //Extract mean and variance at each edge vertex
    T mean0 = inputMeans.Get(edgeIds[0]);
    T mean1 = inputMeans.Get(edgeIds[1]);
    T variance0 = inputVariance.Get(edgeIds[0]);
    T variance1 = inputVariance.Get(edgeIds[1]);

    // Call a function to compute the inverse linear interpolation uncertainty
    gaussian_distribution g;
    viskores::Float64 cross_prob;
    viskores::Float64 var;

#ifdef MAGIC_USE_CLOSED_FORM_ILERP
    // Closed-form computation.
    g.gaussian_alpha_pdf(mean0,
                         variance0,
                         mean1,
                         variance1,
                         0,
                         interpMean,
                         &outputExpectedCrossing,
                         &cross_prob,
                         &var);
#else
    // Monte Carlo computation
    viskores::Id numSamples = 4000;
    g.gaussian_alpha_pdf_MonteCarlo_v2(mean0,
                                       variance0,
                                       mean1,
                                       variance1,
                                       0,
                                       interpMean,
                                       &outputExpectedCrossing,
                                       &cross_prob,
                                       &var,
                                       numSamples);
#endif

    // Uncomment below for the computation of difference between the Monte Carlo and closed-form solutions.
    // You will need to use the above Monte Carlo code too.
    /*viskores::Float64 varTemp;
    g.gaussian_alpha_pdf(mean0,
                         variance0,
                         mean1,
                         variance1,
                         0,
                         interpMean,
                         &outputExpectedCrossing,
                         &cross_prob,
                         &varTemp);
    var = viskores::Abs(var-varTemp);*/

    outputEdgeVariance = var;
  }
};

/* Function to call isosurface uncertainty worklet for a grid edge.
"variance" (input): Array handle representing the variance field (variance/uncertainty in the mean values),
"outputMeanArrayUnknown" (input): Array handle representing same isovalue for a worklet,
"inputMeanArrayUnknown" (input): Array handle representing  the mean field,
"edgeIdsUnknown" (input): Array handle of Ids of edges estimated to be crossed by the isosurface under uncertainty,
"outputEdgeVariance" (output): Array handle represnting variance of isosurface crossing on a grid edge,
"expectedCrossing" (output): Array handle representing expected crossing position of isosurface on a grid edge.
 */
template <typename VarianceArrayType, typename EdgeVarianceArrayType>
VISKORES_CONT void ComputeEdgeVariance(
  const VarianceArrayType& variance,
  const viskores::cont::UnknownArrayHandle& outputMeanArrayUnknown,
  const viskores::cont::UnknownArrayHandle& inputMeanArrayUnknown,
  const viskores::cont::UnknownArrayHandle& edgeIdsUnknown,
  EdgeVarianceArrayType& outputEdgeVariance,
  viskores::cont::ArrayHandle<viskores::Float64>& expectedCrossing)
{
  VarianceArrayType outputMeanArray;
  viskores::cont::ArrayCopyShallowIfPossible(outputMeanArrayUnknown, outputMeanArray);
  VarianceArrayType inputMeanArray;
  viskores::cont::ArrayCopyShallowIfPossible(inputMeanArrayUnknown, inputMeanArray);
  
  viskores::cont::ArrayHandle<viskores::Id2> edgeIds;
  edgeIdsUnknown.AsArrayHandle(edgeIds);

  viskores::cont::Invoker invoke;
  invoke(ComputeEdgeVarianceWorklet{},
         edgeIds,
         outputMeanArray,
         inputMeanArray,
         variance,
         outputEdgeVariance,
         expectedCrossing);

  // Quantitative evaluation: print sum of the differece of Monte Carlo and closed across all grid points
  /*viskores::FloatDefault sum =
    viskores::cont::Algorithm::Reduce(outputEdgeVariance, 0.0f, viskores::Add());
  std::cout<<"Difference in Monte Carlo and Closed-form variance:"<<sum<<"\n";*/
}

/* Worklet to incorporate expected crossing positions (computed with the MAGIC worklet) for isosurface visualization.
 "edgeIds" (input): Ids of edges estimated to be crossed by the isosurface under uncertainty,
 "crossings" (input): Expected crossing position of isosurface on a grid edge computed using MAGIC [0,1],
 "inputArray" (input): Mean field,
 "interpolated" (output): isourface vertex physical position based on expected crossing.
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
"input" (input): Input data with two fields, namely, the mean and variance,
"outputData" (output): Output data of isourface expected vertex positions and their spatial variance.
*/
viskores::cont::DataSet MAGIC::DoExecute(const viskores::cont::DataSet& input)
{
  // Do the actual contour extract while saving the edge information.
  viskores::filter::contour::Contour contourExtract;
  contourExtract.SetIsoValues(this->IsoValues);
  contourExtract.SetGenerateNormals(this->GetGenerateNormals());
  contourExtract.SetComputeFastNormals(this->GetComputeFastNormals());
  contourExtract.SetNormalArrayName(this->GetNormalArrayName());
  contourExtract.SetMergeDuplicatePoints(this->GetMergeDuplicatePoints());
  contourExtract.SetActiveField(0, this->GetActiveFieldName(0), this->GetActiveFieldAssociation());

  // Select fields to pass to the contour filter.
  viskores::filter::FieldSelection fieldSelection = this->GetFieldsToPass();
  for (viskores::IdComponent fieldId = 0; fieldId < input.GetNumberOfFields(); ++fieldId)
  {
    viskores::cont::Field field = input.GetField(fieldId);
    if (field.IsPointField())
    {
      // Do not interpolate point fields as they will have to be reinterpolated later.
      fieldSelection.AddField(field, viskores::filter::FieldSelection::Mode::Exclude);
    }
  }
  
  // Use the mean information to extract the isosurface topology (since it is most probable for the Gaussian model) .
  fieldSelection.AddField(this->GetActiveFieldName(0),
                          this->GetActiveFieldAssociation(0),
                          viskores::filter::FieldSelection::Mode::Select);
  contourExtract.SetFieldsToPass(fieldSelection);

  // Save edge information representing IDs of edges crossed by the isosurface
  contourExtract.SetAddInterpolationEdgeIds(true);

  // Extract the isosurface
  viskores::cont::DataSet contours = contourExtract.Execute(input);

  // Pass array handeles representing the variance field, isovalue, mean field, edge ID (crossed by the isosurface) to compute isosurface expected position and variance on grid edgs using MAGIC.
  viskores::cont::UnknownArrayHandle outputVariance;
  viskores::cont::ArrayHandle<viskores::Float64> expectedCrossings;
  auto resolveArray = [&](auto inputVarianceArray)
  {
    using T = typename std::decay_t<decltype(inputVarianceArray)>::ValueType;
    viskores::cont::ArrayHandle<T> variance;
    ComputeEdgeVariance(inputVarianceArray,
                        this->GetFieldFromDataSet(0, contours).GetData(),
                        this->GetFieldFromDataSet(0, input).GetData(),
                        contours.GetField("edgeIds").GetData(),
                        variance,
                        expectedCrossings);
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
  //outputData.AddPointField(this->GetActiveFieldName(1), outputVariance);
  outputData.AddPointField("variance_edge_crossing", outputVariance);

  // Add the expected positions of isosurface vertex positions to the output dataset (computed with MAGIC).
  outputData.AddPointField("expected_edge_crossing", expectedCrossings);

  return outputData;
}
