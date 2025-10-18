/*============================================================================

Title   -  MAGIC: Marching Cubes Isosurface Uncertainty Visualization for Gaussian Uncertain Data with Spatial Correlation
Authors -  Tushar M. Athawale, Kenneth Moreland, David Pugmire, Chris R. Johnson, Paul Rosen, Matthew Norman, Antigoni Georgiadou, and Alireza Entezari
Date    -  Oct 20, 2025
 
This code implements the following previous paper using the Viskores library:

T. Athawale and A. Entezari, "Uncertainty Quantification in Linear Interpolation for Isosurface Extraction,"
in IEEE Transactions on Visualization and Computer Graphics, vol. 19, no. 12, pp. 2723-2732, Dec. 2013,
doi: 10.1109/TVCG.2013.208.
 
This code is distributed under the OSI-approved BSD 3-clause License. See LICENSE.txt for details.
 
============================================================================*/

#include "MCA_INDEPENDENT_UNIFORM.h"

#include <viskores/cont/ArrayCopy.h>
#include <viskores/Math.h>
#include <viskores/worklet/WorkletMapField.h>

namespace
{

/* Worklet to compute isosurface crossing uncertainty on a grid edge (INDEPENDENT UNIFORM worklet).
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
      
    //Extract the mean and width of the uniform distribution at each edge vertex
    T mean0 = inputMeans.Get(edgeIds[0]);
    T mean1 = inputMeans.Get(edgeIds[1]);
    T variance0 = inputVariance.Get(edgeIds[0]);
    T variance1 = inputVariance.Get(edgeIds[1]);

    // Closed-form computation of isosurface crossing variance on a grid edge
    z_density_uniform u;
    u.setNumUniformInKde1(1);
    u.setNumUniformInKde2(1);
    float tempMean0 = mean0;
    float tempMean1 = mean1;
    viskores::Float64 expected_uniform = u.kde_z_pdf_expected(&tempMean0, variance0, &tempMean1, variance1, interpMean);
    viskores::Float64 var = u.kde_z_pdf_variance(&tempMean0, variance0, &tempMean1, variance1, interpMean, expected_uniform);
      
    outputExpectedCrossing = expected_uniform;
    outputEdgeVariance = var;
  }
};

/* Function to call isosurface uncertainty worklet for a grid edge.
"variance" (input): Array handle representing the uniform noise width field (uncertainty in the mean values),
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
  invoke(
    ComputeEdgeVarianceWorklet{}, edgeIds, outputMeanArray, inputMeanArray, variance, outputEdgeVariance, expectedCrossing);
}

/* Worklet to incorporate expected crossing positions (computed with the INDEPENDENT UNIFORM worklet) for isosurface visualization.
 "edgeIds" (input): Ids of edges estimated to be crossed by the isosurface under uncertainty,
 "crossings" (input): Expected crossing position of isosurface on a grid edge computed using INDEPENDENT UNIFORM [0,1],
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

/* Function to call a worklet that uses expected crossing location (computed with the INDEPENDENT UNIFORM worklet) for isosurface visualization.
"inputArray" (input): Array handle of mean field values,
"edgeIdsUnknown" (input): Array handle of Ids of edges estimated to be crossed by the isosurface under uncertainty,
"crossings" (input): Array handle of expected crossing position of isosurface on a grid edge computed using INDEPENDENT UNIFORM [0,1],
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


/* Filter for isosurface uncertainty computation using the INDEPENDENT UNIFORM worklet.
"input" (input): Input data with two fields, namely, the mean and variance,
"outputData" (output): Output data of isourface expected vertex positions and their spatial variance.
*/
viskores::cont::DataSet MCA_INDEPENDENT_UNIFORM::DoExecute(const viskores::cont::DataSet& input)
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

  // Currently need mean information to figure out interpolation weight.
  fieldSelection.AddField(this->GetActiveFieldName(0),
                          this->GetActiveFieldAssociation(0),
                          viskores::filter::FieldSelection::Mode::Select);
  contourExtract.SetFieldsToPass(fieldSelection);

  // Save edge information representing IDs of edges crossed by the isosurface
  contourExtract.SetAddInterpolationEdgeIds(true);

  // Extract the isosurface
  viskores::cont::DataSet contours = contourExtract.Execute(input);

  viskores::cont::UnknownArrayHandle outputVariance;
  viskores::cont::ArrayHandle<viskores::Float64> expectedCrossings;
    
  //Pass array handeles representing the variance field, isovalue, mean field, edge ID (crossed by the isosurface) to compute isosurface expected position and variance on grid edgs using the INDEPENDENT UNIFORM worklet.
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

  // Use expected isosurface crossing locations on grid edges computed with the INDEPENDENT UNIFORM worklet ([0,1]) to compute physical isosurface vertex locations
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

  // Add the variance of isosurface vertex positions to the output dataset (computed with the INDEPENDENT UNIFORM worklet).
  outputData.AddPointField("variance_edge_crossing", outputVariance);

  // Add the expected positions of isosurface vertex positions to the output dataset (computed with INDEPENDENT UNIFORM worklet).
  outputData.AddPointField("expected_edge_crossing", expectedCrossings);

  return outputData;
}
