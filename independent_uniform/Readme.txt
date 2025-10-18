- Build the viskores library freshly using the viskores source code (tested with
  viskores 1.0).

- Create a build directory where you want to build the marching cubes algorithm (MCA)
  with UNIFORM INDEPENDENT noise code and "cd" to it.

- ccmake source_dir

- For the viskores dir, point to viskoresBuild/lib/cmake/viskores-1.0

- Generate the makefile and create an executable.

- Run "./run-contour data/tangle-independent-uniform.vtk 27.6" 
  (The tangle-independent-uniform.vtk has two fields mean and variance 
  (denoting uniform width) for the independent uniform model. 
  Run with the isovalue 27.6.)

- Visualize the generated output.vtk result file in ParaView colored by
  "variance_edge_crossing" field with inverted "Black-Body-Radiation" colormap
  or a colormap of your choice. Additional ParaView Settings: Ambient light 0.3, 
  Compute surface normals for visualization.