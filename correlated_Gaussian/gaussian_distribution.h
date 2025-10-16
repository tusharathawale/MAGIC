/*============================================================================
Title   -  MAGIC: Marching Cubes Isosurface Uncertainty Visualization for Gaussian Uncertain Data with Spatial Correlation
Authors -  Tushar M. Athawale, Kenneth Moreland, David Pugmire, Chris R. Johnson, Paul Rosen, Matthew Norman, Antigoni Georgiadou, and Alireza Entezari
Date    -  Oct 20, 2025
This code is distributed under the OSI-approved BSD 3-clause License. See LICENSE.txt for details.
 
============================================================================*/

#include <viskores/Math.h>
#include <cmath>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <eigen3/Eigen/Dense>
#include "./linalg/ucv_matrix_static_2by2.h"


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



class gaussian_distribution{

	private :
		

	public :
		
        // Closed-form Computation: Compute uncertainty of inverse linear interpolation for a grid edge with vertex random variables X and Y
        VISKORES_EXEC void gaussian_alpha_pdf(viskores::Float64 muX, viskores::Float64 varX, viskores::Float64 muY, viskores::Float64 varY, viskores::Float64 covarXY, viskores::Float64 c, viskores::Float64* expt, viskores::Float64* cross_prob, viskores::Float64* var);
    
        // helper function for gaussian_alpha_pdf
        VISKORES_EXEC void computeAlphaUncertainty(viskores::Float64 muNumerator, viskores::Float64 muDenominator, viskores::Float64 sigNumerator, viskores::Float64 sigDenominator, viskores::Float64 rhoNumDenom, viskores::Float64* expt, viskores::Float64* cross_prob, viskores::Float64* var);
    
        // Ilerp distribution using Monte Carlo Gaussian sampling (using Eigen and Boost libraries)
        VISKORES_EXEC void gaussian_alpha_pdf_MonteCarlo(double muX, double varX, double muY, double varY, double covarXY, double c, double* expt, double* cross_prob, double* var, int nn);
    
        // Ilerp distribution using Monte Carlo Gaussian sampling (using manually implemented linear algebra (linalg) library)
        VISKORES_EXEC void gaussian_alpha_pdf_MonteCarlo_v2(double muX, double varX, double muY, double varY, double covarXY, double c, double* expt, double* cross_prob, double* var, int nn);

        // helper function for gaussian_alpha_pdf: utilization of the Hinkley's derivation
        VISKORES_EXEC viskores::Float64 hinkleyGaussianRatioCorrelated(viskores::Float64 muNumerator, viskores::Float64 muDenominator, viskores::Float64 sigNumerator, viskores::Float64 sigDenominator, viskores::Float64 rhoNumDenom, viskores::Float64 alphaVal);
		
        // helper function for gaussian_alpha_pdf: utilization of the Marsaglia's and Hinkley's derivations
        VISKORES_EXEC viskores::Float64 hinkleyGaussianRatioIndependent(viskores::Float64 muNumerator, viskores::Float64 muDenominator, viskores::Float64 sigNumerator, viskores::Float64 sigDenominator, viskores::Float64 alphaVal);

        // helper function for gaussian_alpha_pdf: normal cumulative distribution
        VISKORES_EXEC viskores::Float64 normalCumulativeDistribution(viskores::Float64 t, viskores::Float64 mu, viskores::Float64 sig);

};


/* Closed-form Computation: Compute uncertainty of inverse linear interpolation for a grid edge with vertex random variables X and Y.
"muX" (input): mean of the Gaussian distribution for random variable X,
"varX" (input): variance of the Gaussian distribution for random variable X,
"muY" (input): mean of the Gaussian distribution for random variable Y,
"varY" (input): variance of the Gaussian distribution for random variable Y,
"covarXY" (input): sampel covariance of random variables X and Y,
"c" (input): Isovalue,
"expt" (output): Expected crossing location of isosurface on a grid edge (for Gaussian distributed uncertain data),
"cross_prob" (output): Crossing probability of isosurface on a grid edge (for Gaussian distributed uncertain data),
"var" (output): Variance (uncertainty) of crossing location of isosurface on a grid edge (for Gaussian distributed uncertain data).
*/
VISKORES_EXEC inline void gaussian_distribution::gaussian_alpha_pdf(viskores::Float64 muX, viskores::Float64 varX, viskores::Float64 muY, viskores::Float64 varY, viskores::Float64 covarXY, viskores::Float64 c, viskores::Float64* expt, viskores::Float64* cross_prob, viskores::Float64* var)
{
	// Standard devition c-X
	viskores::Float64 sigC_X=viskores::Sqrt(varX);

	// Standard deviation Y-X: This can lead to nan if value to square root is negative
    // Avoid square root of negative value (which should be highly unlikely)
    viskores::Float64 sig_Y_X;

    if (varX + varY -2*covarXY > 0)
        sig_Y_X = viskores::Sqrt(varX + varY -2*covarXY);
    else
        sig_Y_X = viskores::Sqrt(0.00000001);

	// Compute correlation between c-X and Y-X
	viskores::Float64 pearsonCorrelation = (varX - covarXY)/(sigC_X*sig_Y_X);

	// The Pearson's correlation can go above 1 because of numerical instability if there is perfect correlation between X and Y
	if (pearsonCorrelation > 1)
		pearsonCorrelation = 0.99;

	if (pearsonCorrelation < -1)
		pearsonCorrelation = -0.99;
	
	// compute expected value, crossing probability, and variance of alpha distribution using the Hinkley's approach
	computeAlphaUncertainty(c - muX, muY - muX, sigC_X, sig_Y_X, pearsonCorrelation, expt, cross_prob, var);
}


/* Compute uncertainty of inverse linear interpolation (isosurface vertex position), i.e, (isoval - X)/(Y-X) on a grid edge for Gaussian distributed random variables X and Y.
"muNumerator" (input): mean of the Gaussian distribution [G1 for N = isoval-X] at numerator,
"muDenominator" (input): mean of the Gaussian distribution [G2 for D = Y-X] at denominator,
"sigNumerator" (input): Standard deviation of the Gaussian distribution (G1) at numerator,
"sigDenominator" (input): Standard deviation of the Gaussian distribution (G2) at denominator,
"rhoNumDenom" (input): Pearson's correlation between numerator (G1) and denominator (G2) Gaussian random variables,
"expt" (output): Expected crossing location of isosurface on a grid edge (for Gaussian distributed uncertain data),
"cross_prob" (output): Crossing probability of isosurface on a grid edge (for Gaussian distributed uncertain data),
"var" (output): Variance (uncertainty) of crossing location of isosurface on a grid edge (for Gaussian distributed uncertain data).
*/
VISKORES_EXEC inline void gaussian_distribution::computeAlphaUncertainty(viskores::Float64 muNumerator, viskores::Float64 muDenominator, viskores::Float64 sigNumerator, viskores::Float64 sigDenominator, viskores::Float64 rhoNumDenom, viskores::Float64* expt, viskores::Float64* cross_prob, viskores::Float64* var)
{

	// Compute alpha probability at 100 points
	constexpr viskores::IdComponent NUM_POINTS = 100;
	viskores::Float64* alphaDensity = new viskores::Float64[NUM_POINTS];
	viskores::Float64 alphaSum = 0;	
	viskores::Float64 alphaExpected = 0;
	viskores::Float64 alphaVar = 0;	
	
    // Compute inverse linear interpolation random variable density at 100 equispaced locations between 0 and 1.
	for (viskores::IdComponent i=0; i<NUM_POINTS; i++)
	{	
		// we sample only [0,1] range for marching cubes			
		viskores::Float64 alphaVal = (viskores::Float64)i*(1.0/100);
		viskores::Float64 density = hinkleyGaussianRatioCorrelated(muNumerator, muDenominator, sigNumerator, sigDenominator, rhoNumDenom, alphaVal);
		alphaDensity[i] = density;
		// total density over [0,1] for normalization
		alphaSum = alphaSum + density;
	}

    // Compute expected value of isosurface crossing location on grid edge. The domain of crossing position is [0,1].
	for (viskores::IdComponent i=0; i<100; i++)
	{	
		// we sample only [0,1] range for marching cubes			
		viskores::Float64 alphaVal = (viskores::Float64)(i)*(1.0/100);
		// Use normalized density
		alphaExpected = alphaExpected + (viskores::Float64)(alphaDensity[i]/alphaSum)*alphaVal;
	}
	*expt = alphaExpected;
	
    // Compute variance of isosurface crossing location on a grid edge.
	for (viskores::IdComponent i=0; i<100; i++)
	{	
		// we sample only [0,1] range for marching cubes			
		viskores::Float64 alphaVal = (viskores::Float64)(i)*(1.0/100);
		alphaVar = alphaVar + (viskores::Float64)(alphaDensity[i]/alphaSum)*viskores::Pow((alphaVal - *expt),2);
	}
	*expt = alphaExpected;
	*cross_prob = alphaSum;
	*var = alphaVar;
}

/* Monte Carlo Computation (using Eigen and Boost libraries): Compute uncertainty of inverse linear interpolation for a grid edge with vertex random variables X and Y
"muX" (input): mean of the Gaussian distribution for random variable X,
"varX" (input): variance of the Gaussian distribution for random variable X,
"muY" (input): mean of the Gaussian distribution for random variable Y,
"varY" (input): variance of the Gaussian distribution for random variable Y,
"covarXY" (input): sampel covariance of random variables X and Y,
"c" (input): Isovalue,
"expt" (output): Expected crossing location of isosurface on a grid edge (for Gaussian distributed uncertain data),
"cross_prob" (output): Crossing probability of isosurface on a grid edge (for Gaussian distributed uncertain data),
"var" (output): Variance (uncertainty) of crossing location of isosurface on a grid edge (for Gaussian distributed uncertain data),
"nn" (input): Number of Monte Carlo samples.
*/
VISKORES_EXEC inline void gaussian_distribution::gaussian_alpha_pdf_MonteCarlo(double muX, double varX, double muY, double varY, double covarXY, double c, double* expt, double* cross_prob, double* var, int nn)
{

  int size = 2; // Dimensionality (rows)
  //int nn=10;     // How many samples (columns) to draw
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
                   * eigenSolver.eigenvalues().cwiseSqrt().asDiagonal();
  }

  Eigen::MatrixXd samples = (normTransform
                           * Eigen::MatrixXd::NullaryExpr(size,nn,randN)).colwise()
                           + mean;

  // Calculate inverse linear interpolation
  std::vector<double> alphaArr;
  for (int i=0; i<nn; i++)
  {
    double alphaVal;

    // Avoid division by 0
    if (abs(samples.coeff(1,i)-samples.coeff(0,i)) >= 0.001)
    {
        alphaVal = (double)(c-samples.coeff(0,i))/(samples.coeff(1,i)-samples.coeff(0,i));

        // Consider [0,1] domain
        if ((alphaVal >= 0) && (alphaVal<=1))
        {
            alphaArr.push_back(alphaVal);
        }
      }
  }

  //Derive a histogram of inverse linear interpolation values for all samples
  std::sort(alphaArr.begin(), alphaArr.end());

  std::map<double, int> histogram;

  double bin = 0; //Choose your starting bin
  const double bin_width = 0.001; //Choose your bin interval
  for (int e=0; e<alphaArr.size(); e++)
  {
      while (alphaArr[e] >= (bin + bin_width))
        bin += bin_width;
    ++histogram[bin];
  }

  std::map<double, int>::iterator it;
  // Histogram bin centers
  std::vector<double> alphaBin;
  // Histogram bin count
  std::vector<double> alphaBinProb;

  double totalCount = 0;

  for (it=histogram.begin(); it!=histogram.end(); it++)
  {
    alphaBin.push_back(it->first + (double)(bin_width/2.0));
    alphaBinProb.push_back(it->second);
    totalCount+=it->second;
  }

  // Normalize the histogram
  for (int i=0; i<alphaBinProb.size(); i++)
  {
      alphaBinProb[i] = (double)(alphaBinProb[i]/totalCount);
  }

  // Derive expected value of inverse linear interpolation and its variace based on the derived histogram
  double expt_val = 0;
  // Expected value
  for (int i=0; i<alphaBinProb.size(); i++)
  {
      expt_val+=alphaBinProb[i]*alphaBin[i];
  }

  double var_val = 0;
  // Variance
  for (int i=0; i<alphaBinProb.size(); i++)
      var_val+=alphaBinProb[i]*pow(alphaBin[i] - expt_val, 2);

  *expt = expt_val;
  *var = var_val;
  // Crossing probability to be implemented
}

/* Monte Carlo Computation (using manually implemented linear algebra (linalg) library): Compute uncertainty of inverse linear interpolation for a grid edge with vertex random variables X and Y.
"muX" (input): mean of the Gaussian distribution for random variable X,
"varX" (input): variance of the Gaussian distribution for random variable X,
"muY" (input): mean of the Gaussian distribution for random variable Y,
"varY" (input): variance of the Gaussian distribution for random variable Y,
"covarXY" (input): sampel covariance of random variables X and Y,
"c" (input): Isovalue,
"expt" (output): Expected crossing location of isosurface on a grid edge (for Gaussian distributed uncertain data),
"cross_prob" (output): Crossing probability of isosurface on a grid edge (for Gaussian distributed uncertain data),
"var" (output): Variance (uncertainty) of crossing location of isosurface on a grid edge (for Gaussian distributed uncertain data),
"nn" (input): Number of Monte Carlo samples.
*/
VISKORES_EXEC inline void gaussian_distribution::gaussian_alpha_pdf_MonteCarlo_v2(double muX, double varX, double muY, double varY, double covarXY, double c, double* expt, double* cross_prob, double* var, int nn)
{
    
  // Mean vector
  UCVMATH::vec_t ucvmeanv;
  ucvmeanv.v[0] = muX;
  ucvmeanv.v[1] = muY;
    
  // Covariance matrix
  UCVMATH::mat_t ucvcov2by2;
  ucvcov2by2.v[0][0] = varX;
  ucvcov2by2.v[1][1] = varY;
  ucvcov2by2.v[0][1] = covarXY;
  ucvcov2by2.v[1][0] = covarXY;
    
  // Eigenvalue decompostion of covariance matrix
  UCVMATH::mat_t A = UCVMATH::eigen_vector_decomposition(&ucvcov2by2);

  UCVMATH::vec_t sample_v;
  UCVMATH::vec_t AUM;
    
  // Normal distribution random number generator
  std::mt19937 rng;
  rng.seed(std::mt19937::default_seed);
  std::normal_distribution<double> norm;

  // Generate (user-specified number of) Monte Carlo samples and calculate inverse linear interpolation for each sample. Store inverse linear interpolation values for all samples in an array.
  std::vector<double> alphaArr;
  for (int i=0; i<nn; i++)
  {
      
    // get sample vector. Sample two random number and multiply them with eigenvector decomp and add mean
    for (int i = 0; i < 2; i++)
    {
        // using other sample mechanism such as thrust as needed
        sample_v.v[i] = norm(rng);
    }

    AUM = UCVMATH::matrix_mul_vec_add_vec(&A, &sample_v, &ucvmeanv);
      
    double v0_value =  AUM.v[0];
    double v1_value =  AUM.v[1];
      
    double alphaVal;
      
    if (abs(v1_value-v0_value) >= 0.001)
    {
        alphaVal = (double)(c-v0_value)/(v1_value-v0_value);

        // Consider [0,1] domain
        if ((alphaVal >= 0) && (alphaVal<=1))
        {
            alphaArr.push_back(alphaVal);
        }
    }

  }

  //Derive a histogram of inverse linear interpolation values for all samples
  std::sort(alphaArr.begin(), alphaArr.end());

  std::map<double, int> histogram;

  double bin = 0; //Choose your starting bin
  const double bin_width = 0.001; //Choose your bin interval
  for (int e=0; e<alphaArr.size(); e++)
  {
      while (alphaArr[e] >= (bin + bin_width))
        bin += bin_width;
    ++histogram[bin];
  }

  std::map<double, int>::iterator it;
  // Histogram bin centers
  std::vector<double> alphaBin;
  // Histogram bin count
  std::vector<double> alphaBinProb;

  double totalCount = 0;

  for (it=histogram.begin(); it!=histogram.end(); it++)
  {
    alphaBin.push_back(it->first + (double)(bin_width/2.0));
    alphaBinProb.push_back(it->second);
    totalCount+=it->second;
  }

  // Normalize the histogram
  for (int i=0; i<alphaBinProb.size(); i++)
  {
      alphaBinProb[i] = (double)(alphaBinProb[i]/totalCount);
  }

  // Derive the expected value of inverse linear interpolation and its variace based on the derived histogram
  double expt_val = 0;
  // Expected value
  for (int i=0; i<alphaBinProb.size(); i++)
  {
      expt_val+=alphaBinProb[i]*alphaBin[i];
  }

  double var_val = 0;
  // Variance
  for (int i=0; i<alphaBinProb.size(); i++)
      var_val+=alphaBinProb[i]*pow(alphaBin[i] - expt_val, 2);

  *expt = expt_val;
  *var = var_val;
  // Crossing probability to be implemented
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
VISKORES_EXEC inline viskores::Float64 gaussian_distribution::hinkleyGaussianRatioCorrelated(viskores::Float64 muNumerator, viskores::Float64 muDenominator, viskores::Float64 sigNumerator, viskores::Float64 sigDenominator, viskores::Float64 rhoNumDenom, viskores::Float64 alphaVal)
{
	//Compute parameters of transformed random variable that make numertor and denominator uncorrelated
	viskores::Float64 muNumeratorDash = muNumerator - (viskores::Float64)(rhoNumDenom*muDenominator*sigNumerator/sigDenominator);
	viskores::Float64 sigNumeratorDash = sigNumerator*viskores::Sqrt(1.0 - rhoNumDenom*rhoNumDenom);
	viskores::Float64 offset = (viskores::Float64)(rhoNumDenom*sigNumerator/sigDenominator);
	viskores::Float64 alphaDensity = hinkleyGaussianRatioIndependent(muNumeratorDash, muDenominator, sigNumeratorDash, sigDenominator, alphaVal -offset);
	return alphaDensity;
}


/* Compute the ratio of two independent Gaussian distribution functions (G1 and G2) using the derivations from
   Marsaglia's and Hinkley's papers. [G. Marsaglia, Ratios of Normal Variables and Ratios of Sums of Uniform Variables,
   1965] and [D. Hinkley, "On the Ratio of Two Correlated Normal Random Variables", 1969].
// "muNumerator" (input): mean of the Gaussian distribution (G1) at numerator,
// "muDenominator" (input): mean of the Gaussian distribution (G2) at denominator,
// "sigNumerator" (input): Standard deviation of the Gaussian distribution (G1) at numerator,
// "sigDenominator" (input): Standard deviation of the Gaussian distribution (G2) at denominator,
// "alphaVal" (input): Value of the ratio of samples from Gaussian distributions G1 and G2.
*/
VISKORES_EXEC inline viskores::Float64 gaussian_distribution::hinkleyGaussianRatioIndependent(viskores::Float64 muNumerator, viskores::Float64 muDenominator, viskores::Float64 sigNumerator, viskores::Float64 sigDenominator, viskores::Float64 alphaVal)
{
	// if sigNumerator or sigDenominator are zero, a divide by zero error can occur, so add small epsilon
	if (sigNumerator < 0.0001)
		sigNumerator = 0.0001;
	if (sigDenominator < 0.0001)
		sigDenominator = 0.0001;

	viskores::Float64 az = viskores::Sqrt((viskores::Float64)(viskores::Pow(alphaVal,2)/viskores::Pow(sigNumerator,2)) + (viskores::Float64)(1.0/viskores::Pow(sigDenominator,2)));
	viskores::Float64 bz = (viskores::Float64)(muNumerator*alphaVal/viskores::Pow(sigNumerator,2)) + (viskores::Float64)(muDenominator/viskores::Pow(sigDenominator,2));
	viskores::Float64 c = (viskores::Float64)(viskores::Pow(muNumerator,2)/viskores::Pow(sigNumerator,2)) + (viskores::Float64)(viskores::Pow(muDenominator,2)/viskores::Pow(sigDenominator,2));
	viskores::Float64 dz = viskores::Exp((viskores::Pow(bz,2) - c*viskores::Pow(az,2))/(2*viskores::Pow(az,2)));

	viskores::Float64 t1 = normalCumulativeDistribution((viskores::Float64)(bz/az),0,1);
	viskores::Float64 t2 = normalCumulativeDistribution((viskores::Float64)(-bz/az),0,1);

	viskores::Float64 alphaDensity = (viskores::Float64)((bz*dz)/viskores::Pow(az,3))*(viskores::Float64)(1.0/(viskores::Sqrt(2*viskores::Pi())*sigNumerator*sigDenominator))*(t1-t2) + (viskores::Float64)(1.0/(viskores::Pow(az,2)*viskores::Pi()*sigNumerator*sigDenominator)*viskores::Exp((viskores::Float64)(-c/2))); 

	return alphaDensity;
}

/* Compute cumulative distribution function (CDF) of a Gaussian distribution.
"t" (input): Function value,
"mu" (input): Mean of the Gaussian distribution,
"sig" (input): Standard deviation of the Gaussian distribution
*/
VISKORES_EXEC inline viskores::Float64 gaussian_distribution::normalCumulativeDistribution(viskores::Float64 t, viskores::Float64 mu, viskores::Float64 sig)
{
	return 0.5*(1 + std::erf((viskores::Float64)((t-mu)/(viskores::Sqrt(2)*sig))));
}
