##============================================================================
##  Title   -  MAGIC: Marching Cubes Isosurface Uncertainty Visualization for Gaussian Uncertain Data with Spatial Correlation
##  Authors -  Tushar M. Athawale, Kenneth Moreland, David Pugmire, Chris R. Johnson, Paul Rosen, Matthew Norman, Antigoni Georgiadou, and Alireza Entezari
##  Date    -  Oct 20, 2025
##  This code is distributed under the OSI-approved BSD 3-clause License. See LICENSE.txt for details.
##============================================================================

cmake_minimum_required(VERSION 3.23)

project(contour_variance CXX)

find_package(Viskores REQUIRED)
find_package(Boost 1.87 REQUIRED COMPONENTS random)
find_package (Eigen3 3.3 REQUIRED)

add_executable(run-contour run-contour.cxx MCA_KDE_INDEPENDENT_UNIFORM.cxx kdeWithUniformKernel.cxx)

target_link_libraries(run-contour
  PRIVATE viskores::filter_contour viskores::io Boost::random Eigen3::Eigen
  )

viskores_add_target_information(run-contour
  DROP_UNUSED_SYMBOLS
  MODIFY_CUDA_FLAGS
  DEVICE_SOURCES MCA_KDE_INDEPENDENT_UNIFORM.cxx
  )
