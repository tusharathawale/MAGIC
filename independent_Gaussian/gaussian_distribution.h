/*============================================================================

Title   -  MAGIC: Marching Cubes Isosurface Uncertainty Visualization for Gaussian Uncertain Data with Spatial Correlation
Authors -  Tushar M. Athawale, Kenneth Moreland, David Pugmire, Chris R. Johnson, Paul Rosen, Matthew Norman, Antigoni Georgiadou, and Alireza Entezari
Date    -  Oct 20, 2025

This code is distributed under the OSI-approved BSD 3-clause License. See LICENSE.txt for details.
 
============================================================================*/

#include "stdio.h"
#include <iostream>
#include "stdlib.h"
#include "math.h"
#include "./linalg/ucv_matrix_static_2by2.h"

# define infinity 10000
# define minus_infinity -10000

using namespace std;

class gaussian_distribution{

	private :
		

	public :
		
        // Closed-form Computation: Compute uncertainty of inverse linear interpolation for a grid edge with vertex random variables X and Y
        void gaussian_alpha_pdf(double muX, double varX, double muY, double varY, double covarXY, double c, double* expt, double* cross_prob, double* var);

        // helper function for gaussian_alpha_pdf
        void computeAlphaUncertainty(double muNumerator, double muDenominator, double sigNumerator, double sigDenominator, double rhoNumDenom, double* expt, double* cross_prob, double* var);
    
        // Ilerp distribution using Monte Carlo Gaussian sampling (using Eigen and Boost libraries)
        void gaussian_alpha_pdf_MonteCarlo(double muX, double varX, double muY, double varY, double covarXY, double c, double* expt, double* cross_prob, double* var, int numSamples);

        // Ilerp distribution using Monte Carlo Gaussian sampling (using manually implemented linear algebra (linalg) library)
        void gaussian_alpha_pdf_MonteCarlo_v2(double muX, double varX, double muY, double varY, double covarXY, double c, double* expt, double* cross_prob, double* var, int numSamples);

        // helper function for gaussian_alpha_pdf: utilization of the Hinkley's derivation
        double hinkleyGaussianRatioCorrelated(double muNumerator, double muDenominator, double sigNumerator, double sigDenominator, double rhoNumDenom, double alphaVal);
		
        // helper function for gaussian_alpha_pdf: utilization of the Marsaglia's and Hinkley's derivations
        double hinkleyGaussianRatioIndependent(double muNumerator, double muDenominator, double sigNumerator, double sigDenominator, double alphaVal);

        // helper function for gaussian_alpha_pdf: normal cumulative distribution
        double normalCumulativeDistribution(double t, double mu, double sig);
};

