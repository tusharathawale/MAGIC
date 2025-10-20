- Update the line 49 of eigen-based-compute-crossing-probability-entropy.cxx to set the number of ensemble members

- Build viskores freshly using the viskores source code (with serial/OpenMP/GPU backend you need) (tested with viskores 1.0)

- create a build directory where you want to build the crossing probability/entropy (multivariate Gaussian) code and "cd" to it

- ccmake source_dir

- For viskores dir point to viskoresBuild/lib/cmake/viskores-1.0

- Link with the boost and eigen libraries. See CmakeLists.txt in source code

- generate the makefile and and create an executable

- ./eigen-based-compute-crossing-probability-entropy <SyntheticDataSuffix> <FieldName> <Dimx> <Dimy> <Dimz> <num of ensembles> <num of samples> <isovalue> <outputFileSuffix> <eigenThreshold> <true/false (write file or not)>

For example, "./eigen-based-compute-crossing-probability-entropy data/tangle_correlation_one_ tangle 64 64 64 6 200 27.6 tangle-crossing-prob 1 true"

- visualize the output vtk file in ParaView with the crossing_probability/entropy field volume rendered  

- Additional note about the OpenMP usage:

The default Mac clang++/g++compilers do not support OpenMP!! 

So we have to use the llvm clang++ compiler for that purpose.

So update CMAKE_CXX_COMPILER to /opt/homebrew/opt/llvm/bin/clang++ or 
wherever the llvm clang++ is installed