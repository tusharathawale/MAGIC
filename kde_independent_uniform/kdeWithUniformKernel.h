/*============================================================================

Title   -  MAGIC: Marching Cubes Isosurface Uncertainty Visualization for Gaussian Uncertain Data with Spatial Correlation
Authors -  Tushar M. Athawale, Kenneth Moreland, David Pugmire, Chris R. Johnson, Paul Rosen, Matthew Norman, Antigoni Georgiadou, and Alireza Entezari
Date    -  Oct 20, 2025
 
 The following code is courtesy of:

 T. Athawale, E. Sakhaee and A. Entezari, "Isosurface Visualization of Data with Nonparametric Models for Uncertainty,"
 in IEEE Transactions on Visualization and Computer Graphics, vol. 22, no. 1, pp. 777-786, 31 Jan. 2016,
 doi: 10.1109/TVCG.2015.2467958.

This code is distributed under the OSI-approved BSD 3-clause License. See LICENSE.txt for details.
 
============================================================================*/

#include "stdio.h"
#include <iostream>
#include "stdlib.h"
#include "math.h"

# define infinity 10000
# define minus_infinity -10000

typedef struct inverse_quadratic
{  
	// Each piece of form (k1*z^2 + k2*(1-z)^2)/(k3*z^2*(1-z)^2), 
	double k1;
	double k2;
	double k3;
}piece;

typedef struct fn
{         
	// number of pieces in a piecewise function
	int numPieces;
	// maximum number of pieces supported(10)
	piece pc[100];
	// store limits  
	double limits[102]; 

}piecewise;

using namespace std;

class z_density_uniform{

	private :
		double slope_OP, slope_OQ, slope_OR, slope_OS, f1, f2, f3, f4, denominator, mean_temp, delta_temp, numUniformsInKde1, numUniformsInKde2;

	public :
		// Provide piece type(from 0,1,2), actual piece coefficients calculated using alpha_pdf, and limits a(lower) and b(higher) of the integration
		double getSecondMoment(double k1, double k2, double k3, double a, double b);

		// Provide piece type(from 0,1,2), actual piece coefficients calculated using alpha_pdf, and limits a(lower) and b(higher) of the integration
		double getEdgeCrossingProbability(double k1, double k2, double k3, double a, double b);

		// Provide piece type(from 0,1,2), actual piece coefficients calculated using alpha_pdf, and limits a(lower) and b(higher) of the integration
		double getExpectedValOverSinglePiece(double k1, double k2, double k3, double a, double b);

		// p is pdf over -infinity to infinity range. Function returns part of pdf just over range 0 and 1.
		piecewise getPdfOver0To1(piecewise p);

		// Computes expected value, crossing probability and variance over [0,1]
		void Compute0To1(piecewise p, double* expt, double* cross_prob, double* var, double *second_moment, double *first_moment);

		// set number of uniform distributions in kde_1 
		void setNumUniformsInKde1(int a);

		// get number of uniform distributions in kde_1 
		int getNumUniformsInKde1();

		// set number of uniform distributions in kde_2 
		void setNumUniformsInKde2(int a);

		// get number of uniform distributions in kde_2
		int getNumUniformsInKde2();

		// get number of uniform distributions in kde_2
		piecewise getKdePdf(piecewise* p);

		// use k1 only if piece is of type 1 or 2
		piecewise setPiece(piecewise P, int pieceIndex, double k1, double k2);

		// If mu1 > mu2, adjust piece limits
		piecewise adjustPieceLimits(piecewise p);

		// Piecewise function returned assuming data is sampled from a kernel density estimation
		piecewise kde_alpha_pdf(float* mu1, float* delta1, float* mu2, float* delta2, double c);

		// Expected value returned assuming data is sampled from a kernel density estimation
		double kde_alpha_pdf_expected(float* mu1, double h1, float* mu2, double h2, double c);

		// Variance value returned assuming data is sampled from a kernel density estimation
		double kde_alpha_pdf_variance(float* mu1, double h1, float* mu2, double h2, double c);
		
		// Adds 2 piecewise functions   
		piecewise addTwoPiecewiseFunctions(piecewise P1, piecewise P2);

		// add pieces in the range [low high]		
		piece add(piecewise P1, piecewise P2, double low, double high);

		// 1) Non-overlapping Intervals
		piecewise nonOverlapping(double mu1, double delta1, double mu2, double delta2, double c);

		// 2) Overlapping Intervals
		piecewise overlapping(double mu1, double delta1, double mu2, double delta2, double c);

		// 3a) Contained Intervals
		piecewise containedA(double mu1, double delta1, double mu2, double delta2, double c); 

		// 3b) Contained Intervals
		piecewise containedB(double mu1, double delta1, double mu2, double delta2, double c);

		// Piecewise function returned assuming data is sampled from uniform distribution
		piecewise alpha_pdf(double mu1, double delta1, double mu2, double delta2, double c);

};

