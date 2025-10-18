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

using namespace std;

class z_density_uniform{

	private :
		double numUniformInKde1, numUniformInKde2;
		
	public :

    // Isosurface crossing uncertainty on edge for a single pair of uniform distributions
   double alpha_density_uniform(double m1, double d1, double  m2, double  d2, double isovalue, double*  exp, double*  var, double*  crossprob);

    // Expected isosurface crossing position: Kernel density estimation for pairs of uniform distributions
	double kde_z_pdf_expected(float* mu1, double h1, float* mu2, double h2, double c);
	
    // Expected isosurface crossing position for nonlocal means: weighted kernels
	double kde_z_pdf_expected_NL(float* mu1, float* mu1_wts, double h1, float* mu2, float* mu2_wts, double h2, double c);

    // Variance of isosurface crossing position: Kernel density estimation for pairs of uniform distributions
	double kde_z_pdf_variance(float* mu1, double h1, float* mu2, double h2, double c, double expected_crossing);
	
    // Variance of isosurface crossing position for nonlocal means: weighted kernels
	double kde_z_pdf_variance_NL(float* mu1, float* mu1_wts, double h1, float* mu2, float* mu2_wts, double h2, double c, double expected_crossing);
	
	// set number of uniform distributions in kde_1 
	void setNumUniformInKde1(int a);

	// get number of uniform distributions in kde_1 
	int getNumUniformInKde1();

	// set number of uniform distributions in kde_2 
	void setNumUniformInKde2(int a);

	// get number of uniform distributions in kde_2
	int getNumUniformInKde2();
		
};

