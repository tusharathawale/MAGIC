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

# define infinity 10000
# define minus_infinity -10000

using namespace std;

class gaussian_distribution{

	private :
		double numGaussiansInKde1, numGaussiansInKde2;

	public :
    
        // Kernel density estimation for all pairs of kernels for a grid edge
        void kde_gaussian_alpha_pdf(float* mu1, float* delta1, float* mu2, float* delta2, float covarXY, double c, double* expt, double* c_prob, double* var, int numSamples, int option);
    
        // Set the number of Gaussian kernels in kde_1
        void setNumGaussiansInKde1(int a);
    
        // Get the number of Gaussian kernels in kde_1
        int getNumGaussiansInKde1();

        // Set the number of Gaussian kernels in kde_2
        void setNumGaussiansInKde2(int a);

        // Get the number of Gaussian kernels in kde_2
        int getNumGaussiansInKde2();
		
        // Ilerp distribution for a single pair of kernels using closed-form computation
        void gaussian_alpha_pdf(float muX, float varX, float muY, float varY, float covarXY, double c, double* alphaDensity);

        // Helper function for gaussian_alpha_pdf
        void computeAlphaUncertainty(double muNumerator, double muDenominator, double sigNumerator, double sigDenominator, double rhoNumDenom, double* alphaDensity);

		// Helper function for gaussian_alpha_pdf: utilization of the Hinkley's derivation
		double hinkleyGaussianRatioCorrelated(double muNumerator, double muDenominator, double sigNumerator, double sigDenominator, double rhoNumDenom, double alphaVal);
		
		// Helper function for gaussian_alpha_pdf: utilization of the Marsaglia's and Hinkley's derivations
		double hinkleyGaussianRatioIndependent(double muNumerator, double muDenominator, double sigNumerator, double sigDenominator, double alphaVal);

		// Helper function for gaussian_alpha_pdf: normal cumulative distribution
		double normalCumulativeDistribution(double t, double mu, double sig);
    
        // Ilerp distribution using Monte Carlo Gaussian sampling
        void gaussian_alpha_pdf_MonteCarlo(float muX, float varX, float muY, float varY, float covarXY, double c, int numSamples, double* alphaDensity);
};

