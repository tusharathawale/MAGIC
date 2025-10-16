- Edit MAGIC.cxx to select between the closed-form or Monte Carlo versions of
  the inverse linear interpolation (ilerp) uncertainty function by commenting or
  uncommenting the MAGIC_USE_CLOSED_FORM_ILERP macro on line 21.

- Build the viskores library freshly using the viskores source code (tested with
  viskores 1.0).

- Create a build directory where you want to build the MAGIC code and "cd" to
  it.

- ccmake source_dir

- For the viskores dir, point to viskoresBuild/lib/cmake/viskores-1.0

- Install the Boost and Eigen libraries (brew install boost, brew install eigen)
  depending on the version of Monte Carlo function you are using. (Two versions
  are provided in gaussian_distribution.cxx. One (the function
  gaussian_alpha_pdf_MonteCarlo) requires the boost and Eigen libraries, the
  other (the function gaussian_alpha_pdf_MonteCarlo_v2) doesn't require. By
  default the version that does not require these libraries is used.).

- Link with the Boost and Eigen libraries. See CMakeLists.txt in source code.
  Update it if the Boost and Eigen libraries are not required.

- Generate the makefile and create an executable.

- Run "./run-contour data/tangle-correlated-Gaussian-model.vtk 27.6" 
  (The tangle-correlated-Gaussian-model.vtk has five fields mean, variance, rhoX, rhoY, rhoZ  for the correlated Gaussian model. Run with the isovalue 27.6.)

- Visualize the generated output.vtk result file in ParaView colored by
  "variance_edge_crossing" field with inverted "Black-Body-Radiation" colormap
  or a colormap of your choice.