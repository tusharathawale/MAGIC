- Update the line 44 of compute-crossing-probability-entropy.cxx to set the number of ensemble members

- Build viskores freshly using the viskores source code (with serial/OpenMP/GPU backend you need) (tested with viskores 1.0)

- create a build directory where you want to build the crossing probability/entropy (multivariate Gaussian) code and "cd" to it

- ccmake source_dir

- For viskores dir point to viskoresBuild/lib/cmake/viskores-1.0

- Link with the boost and eigen libraries. See CmakeLists.txt in source code

- generate the makefile and and create an executable

- ./crossing-probability-entropy <SyntheticDataSuffix> <FieldName> <Dimx> <Dimy> <Dimz> <num of ensembles> <num of samples> <isovalue> 

For example, "./crossing-probability-entropy data/tangle_correlation_one_ tangle 64 64 64 8 200 27.6"

- visualize the output vtk file in ParaView with the crossing_probability field volume rendered  