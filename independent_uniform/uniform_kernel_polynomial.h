/*============================================================================

Title   -  MAGIC: Marching Cubes Isosurface Uncertainty Visualization for Gaussian Uncertain Data with Spatial Correlation
Authors -  Tushar M. Athawale, Kenneth Moreland, David Pugmire, Chris R. Johnson, Paul Rosen, Matthew Norman, Antigoni Georgiadou, and Alireza Entezari
Date    -  Oct 20, 2025
 
The following code is courtesy of:

T. Athawale and A. Entezari, "Uncertainty Quantification in Linear Interpolation for Isosurface Extraction,"
in IEEE Transactions on Visualization and Computer Graphics, vol. 19, no. 12, pp. 2723-2732, Dec. 2013,
doi: 10.1109/TVCG.2013.208.
 
This code is distributed under the OSI-approved BSD 3-clause License. See LICENSE.txt for details.
 
============================================================================*/

#include "stdio.h"
#include <iostream>
#include "stdlib.h"
#include "math.h"

using namespace std;

class uniform_kernel_polynomial{

	private: double mu1, delta1, mu2, delta2, c;

	public:
		uniform_kernel_polynomial(double m1,double d1,double m2,double d2,double isoval)
		{
			mu1 = m1;
			delta1 = d1;
			mu2 = m2;
			delta2 = d2;
			c = isoval;
		}

		double PQRS_integrate_piece_value(double z, int piecenum);
		double PQRS_expected_piece_value(double z, int piecenum);
		double PQRS_second_moment_piece_value(double z, int piecenum);

		double approx_PQRS_integrate_piece_value(double z, int piecenum);
		double approx_PQRS_expected_piece_value(double z, int piecenum);
		double approx_PQRS_second_moment_piece_value(double z, int piecenum);
};

