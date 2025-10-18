/*============================================================================
 
Title   -  MAGIC: Marching Cubes Isosurface Uncertainty Visualization for Gaussian Uncertain Data with Spatial Correlation
Authors -  Tushar M. Athawale, Kenneth Moreland, David Pugmire, Chris R. Johnson, Paul Rosen, Matthew Norman, Antigoni Georgiadou, and Alireza Entezari
Date    -  Oct 20, 2025
This code is distributed under the OSI-approved BSD 3-clause License. See LICENSE.txt for details.

============================================================================*/

#include <boost/random/normal_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Eigen>

#include <vector>
#include <map>
#include <string>
#include <cmath>
#include "stdio.h"
#include <iostream>
#include "stdlib.h"
#include "math.h"
#include "gaussian_distribution.h"

# define infinity 10000
# define minus_infinity -10000
#define _USE_MATH_DEFINES

/*
  We need a functor that can pretend it's const,
  but to be a good random number generator 
  it needs mutable state.
*/
namespace Eigen {
namespace internal {
template<typename Scalar> 
struct scalar_normal_dist_op 
{
  static boost::mt19937 rng;    // The uniform pseudo-random algorithm
  mutable boost::normal_distribution<Scalar> norm;  // The gaussian combinator

  EIGEN_EMPTY_STRUCT_CTOR(scalar_normal_dist_op)

  template<typename Index>
  inline const Scalar operator() (Index, Index = 0) const { return norm(rng); }
};

template<typename Scalar> boost::mt19937 scalar_normal_dist_op<Scalar>::rng;

template<typename Scalar>
struct functor_traits<scalar_normal_dist_op<Scalar> >
{ enum { Cost = 50 * NumTraits<Scalar>::MulCost, PacketAccess = false, IsRepeatable = false }; };
} // end namespace internal
} // end namespace Eigen


using namespace std;


/* Kernel density estimation to compute isosurface uncertainty.
"mu1" (input): mean of the Gaussian distribution for random variable X,
"dalta1" (input): variance of the Gaussian distribution for random variable X,
"mu2" (input): mean of the Gaussian distribution for random variable Y,
"delta2" (input): variance of the Gaussian distribution for random variable Y,
"covarXY" (input): sampel covariance of random variables X and Y,
"c" (input): Isovalue,
"expt" (output): Expected crossing location of isosurface on a grid edge (for Gaussian distributed uncertain data),
"cross_prob" (output): Crossing probability of isosurface on a grid edge (for Gaussian distributed uncertain data),
"var" (output): Variance (uncertainty) of crossing location of isosurface on a grid edge (for Gaussian distributed uncertain data),
"numSamples" (input): Number of samples to be used if Monte Carlo computation is used,
"option" (input): option=0 for Monte Carlo computation, option=1 for closed-form computation.
*/
void gaussian_distribution :: kde_gaussian_alpha_pdf(float* mu1, float* delta1, float* mu2, float* delta2, float covarXY, double c, double* expt, double* c_prob, double* var, int numSamples, int option)
{
    // Number of ensemble members/kernels at each end of the edge
    int numKde1 = getNumGaussiansInKde1();
    int numKde2 = getNumGaussiansInKde2();

    // Variables to compute the inverse linear interpolation probability at 100 points with array intialized to 0
    double* alphaDensity = new double[100];
    
    // Compute sum for normalization
    for (int i=0; i<100; i++)
    {
        alphaDensity[i] = 0;
    }

    double alphaSum = 0;
    double alphaExpected = 0;
    double alphaVar = 0;
    
    // Loop through all pairs of kernels
    for(int i=0; i<numKde1; i++)
    {
        for(int j=0; j<numKde2; j++)
        {
            // Monte Carlo computation
            if (option == 0)
            {
                double minVal;
                double minDelta;
                double maxVal;
                double maxDelta;

                if (mu1[i] < mu2[j])
                {
                        minVal = mu1[i];
                        minDelta = *delta1;
                        maxVal = mu2[j];
                        maxDelta = *delta2;
                }
                else
                {
                        minVal = mu2[j];
                        minDelta = *delta2;
                        maxVal = mu1[i];
                        maxDelta = *delta1;
                }
                
                if((c > minVal-minDelta) && (c < maxVal+maxDelta))
                {
                    gaussian_alpha_pdf_MonteCarlo(mu1[i], *delta1, mu2[j], *delta2, covarXY, c, numSamples, alphaDensity);
                }

            }
            // Closed-form computation
            else if (option == 1)
                gaussian_alpha_pdf(mu1[i], *delta1, mu2[j], *delta2, covarXY, c, alphaDensity);
        }
    }

    // Compute sum for normalization
    for (int i=0; i<100; i++)
    {
        // total density over [0,1] for normalization
        alphaSum = alphaSum + alphaDensity[i];
    }

    // normalize the alpha density array
    // Compute the expected value
    for (int i=0; i<100; i++)
    {
        // we sample only [0,1] range for marching cubes
        double alphaVal = (double)(i)*(1.0/100);
        // Use normalized density
        alphaExpected = alphaExpected + (double)(alphaDensity[i]/alphaSum)*alphaVal;
    }
    *expt = alphaExpected;
    
    // Compute variance
    for (int i=0; i<100; i++)
    {
        // we sample only [0,1] range for marching cubes
        double alphaVal = (double)(i)*(1.0/100);
        alphaVar = alphaVar + (double)(alphaDensity[i]/alphaSum)*pow((alphaVal - *expt),2);
    }

    *c_prob = alphaSum;
    *var = alphaVar;
    //Crossing probability to be implemented
}

// Set the number of Gaussian kernels in kde1
void gaussian_distribution:: setNumGaussiansInKde1(int a)
{
    numGaussiansInKde1 = a;
}

// Get the number of Gaussian kernels in kde2
int gaussian_distribution:: getNumGaussiansInKde1()
{
    return numGaussiansInKde1;
}

// Set the number of Gaussian kernels in kde1
void gaussian_distribution:: setNumGaussiansInKde2(int a)
{
    numGaussiansInKde2 = a;
}

// Get the number of Gaussian kernels in kde2
int gaussian_distribution:: getNumGaussiansInKde2()
{
    return numGaussiansInKde2;
}

/* Closed-form computation: Compute uncertainty of inverse linear interpolation for a grid edge with vertex random variables X and Y.
"muX" (input): mean of the Gaussian distribution for random variable X,
"varX" (input): variance of the Gaussian distribution for random variable X,
"muY" (input): mean of the Gaussian distribution for random variable Y,
"varY" (input): variance of the Gaussian distribution for random variable Y,
"covarXY" (input): sampel covariance of random variables X and Y,
"c" (input): Isovalue,
"alphaDensity" (output): Array to store histogram of inverse linear interpolation (shared across all pairs of kernels).
*/
void gaussian_distribution::gaussian_alpha_pdf(float muX, float varX, float muY, float varY, float covarXY, double c, double* alphaDensity)
{
	// Standard devition c-X
	double sigC_X=sqrt(varX);

	// Standard deviation Y-X
	double sig_Y_X = sqrt(varX + varY -2*covarXY);

	// Compute correlation between c-X and Y-X
	double pearsonCorrelation = (double)((varX - covarXY)/(sigC_X*sig_Y_X));

	// The Pearson's correlation can go above 1 because of numerical instability if there is perfect correlation between X and Y
	if (pearsonCorrelation > 1)
		pearsonCorrelation = 1;

	if (pearsonCorrelation < -1)
		pearsonCorrelation = -1;
	
	// compute expected value, crossing probability, and variance of alpha distribution using the Hinkley's approach
	computeAlphaUncertainty(c - muX, muY - muX, sigC_X, sig_Y_X, pearsonCorrelation, alphaDensity);
}

/* Compute uncertainty of inverse linear interpolation (isosurface vertex position), i.e., (isoval - X)/(Y-X) on a grid edge for Gaussian distributed random variables X and Y.
"muNumerator" (input): mean of the Gaussian distribution [G1 for N = isoval-X] at numerator,
"muDenominator" (input): mean of the Gaussian distribution [G2 for D = Y-X] at denominator,
"sigNumerator" (input): Standard deviation of the Gaussian distribution (G1) at numerator,
"sigDenominator" (input): Standard deviation of the Gaussian distribution (G2) at denominator,
"rhoNumDenom" (input): Pearson's correlation between numerator (G1) and denominator (G2) Gaussian random variables,
"alphaDensity" (output): Array to store histogram of inverse linear interpolation (shared across all pairs of kernels).
*/
void gaussian_distribution::computeAlphaUncertainty(double muNumerator, double muDenominator, double sigNumerator, double sigDenominator, double rhoNumDenom, double* alphaDensity)
{
    for (int i=0; i<100; i++)
    {
        // we sample only [0,1] range for marching cubes
        double alphaVal = (double)i*(1.0/100);
        double density = hinkleyGaussianRatioCorrelated(muNumerator, muDenominator, sigNumerator, sigDenominator, rhoNumDenom, alphaVal);
        alphaDensity[i] = alphaDensity[i] + density;
    }

}

/* Compute the ratio of two correlated Gaussian distribution functions (G1 and G2) using the derivations from
   Hinkley's paper. [D. Hinkley, "On the Ratio of Two Correlated Normal Random Variables", 1969].
"muNumerator" (input): mean of the Gaussian distribution (G1) at numerator,
"muDenominator" (input): mean of the Gaussian distribution (G2) at denominator,
"sigNumerator" (input): Standard deviation of the Gaussian distribution (G1) at numerator,
"sigDenominator" (input): Standard deviation of the Gaussian distribution (G2) at denominator,
"rhoNumDenom" (input): Pearson's correlation between numerator (G1) and denominator (G2) Gaussian random variables,
"alphaVal" (input): Value of the ratio of samples from Gaussian distributions G1 and G2.
*/
double gaussian_distribution::hinkleyGaussianRatioCorrelated(double muNumerator, double muDenominator, double sigNumerator, double sigDenominator, double rhoNumDenom, double alphaVal)
{
    //Compute parameters of transformed random variable that make numertor and denominator uncorrelated
    double muNumeratorDash = muNumerator - (double)(rhoNumDenom*muDenominator*sigNumerator/sigDenominator);
    double sigNumeratorDash = sigNumerator*sqrt(1.0 - rhoNumDenom*rhoNumDenom);
    double offset = (double)(rhoNumDenom*sigNumerator/sigDenominator);
    double alphaDensity = hinkleyGaussianRatioIndependent(muNumeratorDash, muDenominator, sigNumeratorDash, sigDenominator, alphaVal -offset);
    return alphaDensity;
}

/* Compute the ratio of two independent Gaussian distribution functions (G1 and G2) using the derivations from
   Marsaglia's and Hinkley's papers. [G. Marsaglia, Ratios of Normal Variables and Ratios of Sums of Uniform Variables,
   1965] and [D. Hinkley, "On the Ratio of Two Correlated Normal Random Variables", 1969].
"muNumerator" (input): mean of the Gaussian distribution (G1) at numerator,
"muDenominator" (input): mean of the Gaussian distribution (G2) at denominator,
"sigNumerator" (input): Standard deviation of the Gaussian distribution (G1) at numerator,
"sigDenominator" (input): Standard deviation of the Gaussian distribution (G2) at denominator,
"alphaVal" (input): Value of the ratio of samples from Gaussian distributions G1 and G2.
*/
double gaussian_distribution::hinkleyGaussianRatioIndependent(double muNumerator, double muDenominator, double sigNumerator, double sigDenominator, double alphaVal)
{
    // if sigNumerator or sigDenominator are zero, a divide by zero error can occur, so add small epsilon
    if (sigNumerator < 0.0001)
        sigNumerator = 0.0001;
    if (sigDenominator < 0.0001)
        sigDenominator = 0.0001;

    double az = sqrt((double)(pow(alphaVal,2)/pow(sigNumerator,2)) + (double)(1.0/pow(sigDenominator,2)));
    double bz = (double)(muNumerator*alphaVal/pow(sigNumerator,2)) + (double)(muDenominator/pow(sigDenominator,2));
    double c = (double)(pow(muNumerator,2)/pow(sigNumerator,2)) + (double)(pow(muDenominator,2)/pow(sigDenominator,2));
    double dz = exp((pow(bz,2) - c*pow(az,2))/(2*pow(az,2)));

    double t1 = normalCumulativeDistribution((double)(bz/az),0,1);
    double t2 = normalCumulativeDistribution((double)(-bz/az),0,1);

    double alphaDensity = (double)((bz*dz)/pow(az,3))*(double)(1.0/(sqrt(2*M_PI)*sigNumerator*sigDenominator))*(t1-t2) + (double)(1.0/(pow(az,2)*M_PI*sigNumerator*sigDenominator)*exp((double)(-c/2)));

    return alphaDensity;
}

/* Compute cumulative distribution function (CDF) of a Gaussian distribution.
"t" (input): Function value,
"mu" (input): Mean of the Gaussian distribution,
"sig" (input): Standard deviation of the Gaussian distribution
*/
double gaussian_distribution::normalCumulativeDistribution(double t, double mu, double sig)
{
    return 0.5*(1 + erf((double)((t-mu)/(sqrt(2)*sig))));
}

/* Monte Carlo computation: Compute uncertainty of inverse linear interpolation for a grid edge with vertex random variables X and Y.
"muX" (input): mean of the Gaussian distribution for random variable X,
"varX" (input): variance of the Gaussian distribution for random variable X,
"muY" (input): mean of the Gaussian distribution for random variable Y,
"varY" (input): variance of the Gaussian distribution for random variable Y,
"covarXY" (input): sampel covariance of random variables X and Y,
"c" (input): Isovalue,
"nn" (input): Number of Monte Carlo samples,
"alphaDensity" (output): Array to store histogram of inverse linear interpolation (shared across all pairs of kernels).
*/
void gaussian_distribution::gaussian_alpha_pdf_MonteCarlo(float muX, float varX, float muY, float varY, float covarXY, double c, int nn, double* alphaDensity)
{

  int size = 2; // Dimensionality (rows)
  Eigen::internal::scalar_normal_dist_op<double> randN; // Gaussian functor
  Eigen::internal::scalar_normal_dist_op<double>::rng.seed(1); // Seed the rng

  // Define mean and covariance of the distribution
  Eigen::VectorXd mean(size);
  Eigen::MatrixXd covar(size,size);

  mean  <<  muX,  muY;
  covar <<  varX, covarXY,
           covarXY,  varY;

  Eigen::MatrixXd normTransform(size,size);

  Eigen::LLT<Eigen::MatrixXd> cholSolver(covar);

  // We can only use the cholesky decomposition if
  // the covariance matrix is symmetric, pos-definite.
  // But a covariance matrix might be pos-semi-definite.
  // In that case, we'll go to an EigenSolver
  if (cholSolver.info()==Eigen::Success) {
    // Use cholesky solver
    normTransform = cholSolver.matrixL();
  } else {
    // Use eigen solver
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(covar);
    normTransform = eigenSolver.eigenvectors()
                   * eigenSolver.eigenvalues().cwiseMax(0).cwiseSqrt().asDiagonal();
  }

 // Calculate inverse linear interpolation for each pair of samples
 int iter = 0;
 double alphaVal;
 double* alphaArr = new double[nn];
 while (iter<nn)
 {
    Eigen::MatrixXd samples = (normTransform
                           * Eigen::MatrixXd::NullaryExpr(size,1,randN)).colwise()
                           + mean;
    // Avoid division by 0
    if (abs(samples.coeff(1,0)-samples.coeff(0,0)) >= 0.001)
    {

        alphaVal = (double)(c-samples.coeff(0,0))/(samples.coeff(1,0)-samples.coeff(0,0));

        // Consider [0,1] domain
        if ((alphaVal >= 0) && (alphaVal<=1))
        {
            alphaArr[iter]=alphaVal;
            iter++;
        }
      }
    
 }

 //Derive a histogram of inverse linear interpolation values for all samples
 sort(alphaArr, alphaArr+nn);

 // Fill the inverse linear interpolation density array for a pair of kernels at hand
 double bin = 0; //Choose your starting bin
 const double bin_width = 0.01; //Choose your bin interval
 int alphaDensityId = 0;
 for (int e=0; e<nn; e++)
 {
     while (alphaArr[e] >= (bin + bin_width))
     {
         bin += bin_width;
         alphaDensityId+=1;
     }
     ++alphaDensity[alphaDensityId];
  }

}
