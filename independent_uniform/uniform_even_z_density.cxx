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
#include <algorithm>
#include "uniform_even_z_density.h"
#include <cfloat>
#include <limits> 
#include <sys/time.h>
#include "uniform_kernel_polynomial.h"

# define infinity 10000
# define minus_infinity -10000


#define _USE_MATH_DEFINES

using namespace std;

double get_wall_time(){
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        //  Handle error
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}
double get_cpu_time(){
    return (double)clock() / CLOCKS_PER_SEC;
}


double u_mu1, u_delta1, u_mu2, u_delta2, u_c;


/*

                      b
             r -  -  -  -  -  - q
            /        /        /  
           /   P3   /   P2   /
          /        /  	    /
        c  -  -  - e -  -  - a
        /         /         /
       /    P4   /   P1    /
      /         /         /
      s-  -  -  d- - - - - p

% P1, P2, P3, P4 are different polynomials.*/

uniform_kernel_polynomial* uni_poly;

double u_eps = 1.0e-4;

// Test a>=b
inline int u_ge(double a, double b)
{
	if(a>b)
		return 1;
	if(fabs(a-b) < u_eps)
		return 1;
	return 0;
}

// Test a<=b
inline int u_le(double a, double b)
{
	if(a<b)
		return 1;
	if(fabs(a-b) < u_eps)
		return 1;	
	return 0;
}

// Test a==b
inline int u_eq(double a, double b)
{
	if(fabs(a-b) < u_eps)
		return 1;
	return 0;
}

double u_integrate_pdf(double low, double high, int piecenum)
{
	if(u_ge(high,1))
	{
		high = 1;
	}
	
	if(u_le(low,0))
	{
		low = 0;
	}

	if (u_ge(low,1))
		return 0;
	else if (high - low < u_eps)
		return 0;
	else		
		return (uni_poly -> PQRS_integrate_piece_value(high,piecenum) - uni_poly -> PQRS_integrate_piece_value(low,piecenum));
}

double u_expected_value(double low, double high, int piecenum)
{	
	if(u_ge(high,1))
	{
		high = 1;
	}

	if(u_le(low,0))
	{
		low = 0;
	}
	
	// assuming low is always greater than 0 and  low<=high
	if (u_ge(low,1))
		return 0;
	else if (high - low < u_eps)
		return 0;
	else		
		return uni_poly -> PQRS_expected_piece_value(high,piecenum) - uni_poly -> PQRS_expected_piece_value(low,piecenum);
}

double u_second_moment(double low, double high, int piecenum)
{	

	if(u_ge(high,1))
	{
		high = 1;
	}
	
	if(u_le(low,0))
	{
		low = 0;
	}

	// assuming low is always greater than 0 and  low<=high
	if (u_ge(low,1))
		return 0;
	else if (high - low < u_eps)
		return 0;
	else		
		return uni_poly -> PQRS_second_moment_piece_value(high,piecenum) - uni_poly -> PQRS_second_moment_piece_value(low,piecenum);
}


double u_path_1(double p,double a,double e,double d, double* exp, double* var, double *crossprob)
{    
    
	*exp = 0;
	*var = 0;
	*crossprob = 0;	

	// Vertex Order PADE 
	if (u_le(p,a) && u_le(a,d) && u_le(d,e))
	{   				
	*exp = u_expected_value(p,a,1) + u_expected_value(a,d,5) - u_expected_value(d,e,3);
	*var = u_second_moment(p,a,1) + u_second_moment(a,d,5) - u_second_moment(d,e,3);
	*crossprob = u_integrate_pdf(p,a,1) + u_integrate_pdf(a,d,5) - u_integrate_pdf(d,e,3);
	}
	
	// Vertex Order PAED 
	else if (u_le(p,a) && u_le(a,e) && u_le(e,d))
 	{
	*exp = u_expected_value(p,a,1) + u_expected_value(a,e,5) - u_expected_value(e,d,4);
	*var = u_second_moment(p,a,1) + u_second_moment(a,e,5) - u_second_moment(e,d,4);
	*crossprob = u_integrate_pdf(p,a,1) + u_integrate_pdf(a,e,5) - u_integrate_pdf(e,d,4);	
	}

	// Vertex Order PDAE 
	else if (u_le(p,d) && u_le(d,a) && u_le(a,e))
	{    	    		
	*exp = u_expected_value(p,d,1) + u_expected_value(d,a,6) - u_expected_value(a,e,3);
	*var = u_second_moment(p,d,1) + u_second_moment(d,a,6) - u_second_moment(a,e,3);
	*crossprob = u_integrate_pdf(p,d,1) + u_integrate_pdf(d,a,6) - u_integrate_pdf(a,e,3);
	}
	
}

double u_path_2(double p,double a,double e,double d, double* exp, double* var, double *crossprob)
{
	*exp = 0;
	*var = 0;
	*crossprob = 0;

	// Vertex Order EDAP (polynomial 1)
	if (u_le(e,d) && u_le(d,a) && u_le(a,p))
    	{   	 	
	*exp = u_expected_value(e,d,3) - u_expected_value(d,a,5) - u_expected_value(a,p,1);
	*var = u_second_moment(e,d,3) - u_second_moment(d,a,5) - u_second_moment(a,p,1);
	*crossprob = u_integrate_pdf(e,d,3) - u_integrate_pdf(d,a,5) - u_integrate_pdf(a,p,1);	
	}
    
    	// Vertex Order EDPA (polynomial 1)
	else if (u_le(e,d) && u_le(d,p) && u_le(p,a))
    	{      
	*exp = u_expected_value(e,d,3) - u_expected_value(d,p,5) - u_expected_value(p,a,2);
	*var = u_second_moment(e,d,3) - u_second_moment(d,p,5) - u_second_moment(p,a,2);
	*crossprob = u_integrate_pdf(e,d,3) - u_integrate_pdf(d,p,5) - u_integrate_pdf(p,a,2);
     	}

    	// Vertex Order EADP (polynomial 1)
	else if (u_le(e,a) && u_le(a,d) && u_le(d,p))
    	{    
	*exp = u_expected_value(e,a,3) - u_expected_value(a,d,6) - u_expected_value(d,p,1);
	*var = u_second_moment(e,a,3) - u_second_moment(a,d,6) - u_second_moment(d,p,1);
	*crossprob = u_integrate_pdf(e,a,3) - u_integrate_pdf(a,d,6) - u_integrate_pdf(d,p,1);		
	}	
}

double u_path_3(double p,double a,double e,double d, double* exp, double* var, double *crossprob)
{
	*exp = 0;
	*var = 0;
	*crossprob = 0;

	/* Only 1 possible ordering DPAE
 All equations derived from PAED_positive_1
 first and the fourth part should be identical*/
	*exp = -u_expected_value(0,d,6) - u_expected_value(d,p,1) + u_expected_value(a,e,2) - u_expected_value(e,INFINITY,6);
	*var = -u_second_moment(0,d,6) - u_second_moment(d,p,1) + u_second_moment(a,e,2) - u_second_moment(e,INFINITY,6);
	*crossprob = -u_integrate_pdf(0,d,6) - u_integrate_pdf(d,p,1) + u_integrate_pdf(a,e,2) - u_integrate_pdf(e,INFINITY,6);	
}

double u_path_4(double p,double a,double e,double d, double* exp, double* var, double *crossprob)
{
	*exp = 0;
	*var = 0;
	*crossprob = 0;
	
	/* Only 1 possible ordering AEDP
% All equations derived from PAED_positive_1
% first and the fourth part should be identical*/	
	*exp = u_expected_value(0,a,6) - u_expected_value(a,e,3) + u_expected_value(d,p,4) + u_expected_value(p,INFINITY,6);	
	*var = u_second_moment(0,a,6) - u_second_moment(a,e,3) + u_second_moment(d,p,4) + u_second_moment(p,INFINITY,6);	
	*crossprob = u_integrate_pdf(0,a,6) - u_integrate_pdf(a,e,3) + u_integrate_pdf(d,p,4) + u_integrate_pdf(p,INFINITY,6);	
}

double u_path_5(double p,double a,double e,double d, double* exp, double* var, double *crossprob)
{
	*exp = 0;
	*var = 0;
	*crossprob = 0;
	
	// Vertex Order DAPE 
	if (u_le(d,a) && u_le(a,p) && u_le(p,e))
	{  
	*exp = u_expected_value(0,d,7) + u_expected_value(0,d,8) + u_expected_value(d,a,7) + u_expected_value(d,a,10) + u_expected_value(a,p,9)+ u_expected_value(a,p,10) + u_expected_value(p,e,9)+ u_expected_value(p,e,12) + u_expected_value(e,INFINITY,11)+ u_expected_value(e,INFINITY,12);	
	*var = u_second_moment(0,d,7) + u_second_moment(0,d,8) + u_second_moment(d,a,7) + u_second_moment(d,a,10) + u_second_moment(a,p,9)+ u_second_moment(a,p,10) + u_second_moment(p,e,9)+ u_second_moment(p,e,12) + u_second_moment(e,INFINITY,11)+ u_second_moment(e,INFINITY,12);	
	*crossprob = u_integrate_pdf(0,d,7) + u_integrate_pdf(0,d,8) + u_integrate_pdf(d,a,7) + u_integrate_pdf(d,a,10) + u_integrate_pdf(a,p,9)+ u_integrate_pdf(a,p,10) + u_integrate_pdf(p,e,9)+ u_integrate_pdf(p,e,12) + u_integrate_pdf(e,INFINITY,11)+ u_integrate_pdf(e,INFINITY,12);
	}
    
    	// Vertex Order ADPE (polynomial 1)
	else if (u_le(a,d) && u_le(d,p) && u_le(p,e))
	{
	*exp = u_expected_value(0,a,7) + u_expected_value(0,a,8) + u_expected_value(a,d,9) + u_expected_value(a,d,8) + u_expected_value(d,p,9)+ u_expected_value(d,p,10) + u_expected_value(p,e,9)+ u_expected_value(p,e,12) + u_expected_value(e,INFINITY,11)+ u_expected_value(e,INFINITY,12);
	*var = u_second_moment(0,a,7) + u_second_moment(0,a,8) + u_second_moment(a,d,9) + u_second_moment(a,d,8) + u_second_moment(d,p,9)+ u_second_moment(d,p,10) + u_second_moment(p,e,9)+ u_second_moment(p,e,12) + u_second_moment(e,INFINITY,11)+ u_second_moment(e,INFINITY,12);
	*crossprob = u_integrate_pdf(0,a,7) + u_integrate_pdf(0,a,8) + u_integrate_pdf(a,d,9) + u_integrate_pdf(a,d,8) + u_integrate_pdf(d,p,9)+ u_integrate_pdf(d,p,10) + u_integrate_pdf(p,e,9)+ u_integrate_pdf(p,e,12) + u_integrate_pdf(e,INFINITY,11)+ u_integrate_pdf(e,INFINITY,12);
	}
    
	// Vertex Order DAEP (polynomial 1)
	else if (u_le(d,a) && u_le(a,e) && u_le(e,p))
	{  
	*exp = u_expected_value(0,d,7) + u_expected_value(0,d,8) + u_expected_value(d,a,7) + u_expected_value(d,a,10) + u_expected_value(a,e,9)+ u_expected_value(a,e,10) + u_expected_value(e,p,11)+ u_expected_value(e,p,10) + u_expected_value(p,INFINITY,11)+ u_expected_value(p,INFINITY,12);
	*var = u_second_moment(0,d,7) + u_second_moment(0,d,8) + u_second_moment(d,a,7) + u_second_moment(d,a,10) + u_second_moment(a,e,9)+ u_second_moment(a,e,10) + u_second_moment(e,p,11)+ u_second_moment(e,p,10) + u_second_moment(p,INFINITY,11)+ u_second_moment(p,INFINITY,12);
	*crossprob = u_integrate_pdf(0,d,7) + u_integrate_pdf(0,d,8) + u_integrate_pdf(d,a,7) + u_integrate_pdf(d,a,10) + u_integrate_pdf(a,e,9)+ u_integrate_pdf(a,e,10) + u_integrate_pdf(e,p,11)+ u_integrate_pdf(e,p,10) + u_integrate_pdf(p,INFINITY,11)+ u_integrate_pdf(p,INFINITY,12);
	}
   
   	// Vertex Order ADEP (polynomial 1)
	else if (u_le(a,d) && u_le(d,e) && u_le(e,p))
	{       
	*exp = u_expected_value(0,a,7) + u_expected_value(0,a,8) + u_expected_value(a,d,9) + u_expected_value(a,d,8) + u_expected_value(d,e,9)+ u_expected_value(d,e,10) + u_expected_value(e,p,11)+ u_expected_value(e,p,10) + u_expected_value(p,INFINITY,11)+ u_expected_value(p,INFINITY,12);
	*var = u_second_moment(0,a,7) + u_second_moment(0,a,8) + u_second_moment(a,d,9) + u_second_moment(a,d,8) + u_second_moment(d,e,9)+ u_second_moment(d,e,10) + u_second_moment(e,p,11)+ u_second_moment(e,p,10) + u_second_moment(p,INFINITY,11)+ u_second_moment(p,INFINITY,12);
	*crossprob = u_integrate_pdf(0,a,7) + u_integrate_pdf(0,a,8) + u_integrate_pdf(a,d,9) + u_integrate_pdf(a,d,8) + u_integrate_pdf(d,e,9)+ u_integrate_pdf(d,e,10) + u_integrate_pdf(e,p,11)+ u_integrate_pdf(e,p,10) + u_integrate_pdf(p,INFINITY,11)+ u_integrate_pdf(p,INFINITY,12);
	}	
}

double u_path_6(double p,double a,double e,double d, double* exp, double* var, double *crossprob)
{
	*exp = 0;
	*var = 0;
	*crossprob = 0;

//% only one possible vertex ordering DPA (Ignore -ve vertices)
	*exp = -u_expected_value(0,d,6) - u_expected_value(d,p,1) + u_expected_value(a,INFINITY,2);	
	*var = -u_second_moment(0,d,6) - u_second_moment(d,p,1) + u_second_moment(a,INFINITY,2);	
	*crossprob = -u_integrate_pdf(0,d,6) - u_integrate_pdf(d,p,1) + u_integrate_pdf(a,INFINITY,2);		
}

double u_path_7(double p,double a,double e,double d, double* exp, double* var, double *crossprob)
{
	*exp = 0;
	*var = 0;
	*crossprob = 0;

	// vertex order DAP
	if (u_le(d,a) && u_le(a,p))
	{  	
	*exp = u_expected_value(0,d,7) + u_expected_value(0,d,8) + u_expected_value(d,a,7) + u_expected_value(d,a,10) + u_expected_value(a,p,9)+ u_expected_value(a,p,10) + u_expected_value(p,INFINITY,9)+ u_expected_value(p,INFINITY,12);
	*var = u_second_moment(0,d,7) + u_second_moment(0,d,8) + u_second_moment(d,a,7) + u_second_moment(d,a,10) + u_second_moment(a,p,9)+ u_second_moment(a,p,10) + u_second_moment(p,INFINITY,9)+ u_second_moment(p,INFINITY,12);
	*crossprob = u_integrate_pdf(0,d,7) + u_integrate_pdf(0,d,8) + u_integrate_pdf(d,a,7) + u_integrate_pdf(d,a,10) + u_integrate_pdf(a,p,9)+ u_integrate_pdf(a,p,10) + u_integrate_pdf(p,INFINITY,9)+ u_integrate_pdf(p,INFINITY,12);
	}
    
    	// vertex order ADP
	else if (u_le(a,d) && u_le(d,p))
	{       
	*exp = u_expected_value(0,a,7) + u_expected_value(0,a,8) + u_expected_value(a,d,9) + u_expected_value(a,d,8) + u_expected_value(d,p,9)+ u_expected_value(d,p,10) + u_expected_value(p,INFINITY,9)+ u_expected_value(p,INFINITY,12);
	*var = u_second_moment(0,a,7) + u_second_moment(0,a,8) + u_second_moment(a,d,9) + u_second_moment(a,d,8) + u_second_moment(d,p,9)+ u_second_moment(d,p,10) + u_second_moment(p,INFINITY,9)+ u_second_moment(p,INFINITY,12);
	*crossprob = u_integrate_pdf(0,a,7) + u_integrate_pdf(0,a,8) + u_integrate_pdf(a,d,9) + u_integrate_pdf(a,d,8) + u_integrate_pdf(d,p,9)+ u_integrate_pdf(d,p,10) + u_integrate_pdf(p,INFINITY,9)+ u_integrate_pdf(p,INFINITY,12);        
	}	
}

double u_path_8(double p,double a,double e,double d, double* exp, double* var, double *crossprob)
{
	*exp = 0;
	*var = 0;
	*crossprob = 0;
	
	// only one possible vertex ordering AED (Ignore -ve vertices)
	*exp = u_expected_value(0,a,6) - u_expected_value(a,e,3) + u_expected_value(d,INFINITY,4);	
	*var = u_second_moment(0,a,6) - u_second_moment(a,e,3) + u_second_moment(d,INFINITY,4);	
	*crossprob = u_integrate_pdf(0,a,6) - u_integrate_pdf(a,e,3) + u_integrate_pdf(d,INFINITY,4);		
}

double u_path_9(double p,double a,double e,double d, double* exp, double* var, double *crossprob)
{
	*exp = 0;
	*var = 0;
	*crossprob = 0;
	
	// vertex order ADE
	if (u_le(a,d) && u_le(d,e))
	{       
	*exp = u_expected_value(0,a,7) + u_expected_value(0,a,8) + u_expected_value(a,d,9) + u_expected_value(a,d,8) + u_expected_value(d,e,9)+ u_expected_value(d,e,10) + u_expected_value(e,INFINITY,11)+ u_expected_value(e,INFINITY,10);
	*var = u_second_moment(0,a,7) + u_second_moment(0,a,8) + u_second_moment(a,d,9) + u_second_moment(a,d,8) + u_second_moment(d,e,9)+ u_second_moment(d,e,10) + u_second_moment(e,INFINITY,11)+ u_second_moment(e,INFINITY,10);
	*crossprob = u_integrate_pdf(0,a,7) + u_integrate_pdf(0,a,8) + u_integrate_pdf(a,d,9) + u_integrate_pdf(a,d,8) + u_integrate_pdf(d,e,9)+ u_integrate_pdf(d,e,10) + u_integrate_pdf(e,INFINITY,11)+ u_integrate_pdf(e,INFINITY,10);
	}
    
	// vertex order DAE
	else if (u_le(d,a) && u_le(a,e))
	{    
	*exp = u_expected_value(0,d,7) + u_expected_value(0,d,8) + u_expected_value(d,a,7) + u_expected_value(d,a,10) + u_expected_value(a,e,9)+ u_expected_value(a,e,10) + u_expected_value(e,INFINITY,11)+ u_expected_value(e,INFINITY,10);
	*var = u_second_moment(0,d,7) + u_second_moment(0,d,8) + u_second_moment(d,a,7) + u_second_moment(d,a,10) + u_second_moment(a,e,9)+ u_second_moment(a,e,10) + u_second_moment(e,INFINITY,11)+ u_second_moment(e,INFINITY,10);
	*crossprob = u_integrate_pdf(0,d,7) + u_integrate_pdf(0,d,8) + u_integrate_pdf(d,a,7) + u_integrate_pdf(d,a,10) + u_integrate_pdf(a,e,9)+ u_integrate_pdf(a,e,10) + u_integrate_pdf(e,INFINITY,11)+ u_integrate_pdf(e,INFINITY,10);
	}	
}

double u_path_10(double p,double a,double e,double d, double* exp, double* var, double *crossprob)
{
	*exp = 0;
	*var = 0;
	*crossprob = 0;

	// vertex order PAE
	if (u_le(p,a) && u_le(a,e))
	{ 
	*exp = u_expected_value(p,a,1) + u_expected_value(a,e,5) - u_expected_value(e,INFINITY,4);	
	*var = u_second_moment(p,a,1) + u_second_moment(a,e,5) - u_second_moment(e,INFINITY,4);	
	*crossprob = u_integrate_pdf(p,a,1) + u_integrate_pdf(a,e,5) - u_integrate_pdf(e,INFINITY,4);	
	}
    
	// vertex order APE    
	else if (u_le(a,p) && u_le(p,e)) 
	{    
	*exp = u_expected_value(a,p,2) + u_expected_value(p,e,5) - u_expected_value(e,INFINITY,4);	
	*var = u_second_moment(a,p,2) + u_second_moment(p,e,5) - u_second_moment(e,INFINITY,4);	
	*crossprob = u_integrate_pdf(a,p,2) + u_integrate_pdf(p,e,5) - u_integrate_pdf(e,INFINITY,4);	
	}    

	// vertex order AEP      
	else if (u_le(a,e) && u_le(e,p))  
	{    
	*exp = u_expected_value(a,e,2) - u_expected_value(e,p,6) - u_expected_value(p,INFINITY,4);	
	*var = u_second_moment(a,e,2) - u_second_moment(e,p,6) - u_second_moment(p,INFINITY,4);	
	*crossprob = u_integrate_pdf(a,e,2) - u_integrate_pdf(e,p,6) - u_integrate_pdf(p,INFINITY,4);		
	}	
}

double u_path_11(double p,double a,double e,double d, double* exp, double* var, double *crossprob)
{
	*exp = 0;
	*var = 0;
	*crossprob = 0;
	
	// vertex order EDP
	if (u_le(e,d) && u_le(d,p))
	{  
	*exp = u_expected_value(e,d,3) - u_expected_value(d,p,5) - u_expected_value(p,INFINITY,2);	
	*var = u_second_moment(e,d,3) - u_second_moment(d,p,5) - u_second_moment(p,INFINITY,2);		
	*crossprob = u_integrate_pdf(e,d,3) - u_integrate_pdf(d,p,5) - u_integrate_pdf(p,INFINITY,2);	 		
	}    

	// vertex order DEP    
	else if (u_le(d,e) && u_le(e,p))  
	{     
	*exp = u_expected_value(d,e,4) - u_expected_value(e,p,5) - u_expected_value(p,INFINITY,2);	
	*var = u_second_moment(d,e,4) - u_second_moment(e,p,5) - u_second_moment(p,INFINITY,2);	
	*crossprob = u_integrate_pdf(d,e,4) - u_integrate_pdf(e,p,5) - u_integrate_pdf(p,INFINITY,2);	
	}
    
	// vertex order DPE    
	else if (u_le(d,p) && u_le(p,e))  
	{	         
	*exp = u_expected_value(d,p,4) + u_expected_value(p,e,6) - u_expected_value(e,INFINITY,2);	
	*var = u_second_moment(d,p,4) + u_second_moment(p,e,6) - u_second_moment(e,INFINITY,2);	
	*crossprob = u_integrate_pdf(d,p,4) + u_integrate_pdf(p,e,6) - u_integrate_pdf(e,INFINITY,2);	
	}	
}

double u_path_12(double p,double a,double e,double d, double* exp, double* var, double *crossprob)
{
	*exp = 0;
	*var = 0;
	*crossprob = 0;

	// Vertex order PA
	if (u_le(p,a))
	{  
	*exp = u_expected_value(p,a,1) + u_expected_value(a,INFINITY,5);
	*var = u_second_moment(p,a,1) + u_second_moment(a,INFINITY,5);
	*crossprob = u_integrate_pdf(p,a,1) + u_integrate_pdf(a,INFINITY,5);
	}
    
	// Vertex order AP
	else if (u_le(a,p))
	{      
	*exp = u_expected_value(a,p,2) + u_expected_value(p,INFINITY,5);
	*var = u_second_moment(a,p,2) + u_second_moment(p,INFINITY,5);
	*crossprob = u_integrate_pdf(a,p,2) + u_integrate_pdf(p,INFINITY,5);
	}	
}

double u_path_13(double p,double a,double e,double d, double* exp, double* var, double *crossprob)
{
	*exp = 0;
	*var = 0;
	*crossprob = 0;
	
	// Vertex order ED
	if (u_le(e,d))
	{  
	*exp = u_expected_value(e,d,3) - u_expected_value(d,INFINITY,5);
	*var = u_second_moment(e,d,3) - u_second_moment(d,INFINITY,5);
	*crossprob = u_integrate_pdf(e,d,3) - u_integrate_pdf(d,INFINITY,5);
	}
    
	// Vertex order DE
	else if (u_le(d,e))
	{     
	*exp = u_expected_value(d,e,4) - u_expected_value(e,INFINITY,5);
	*var = u_second_moment(d,e,4) - u_second_moment(e,INFINITY,5);
	*crossprob = u_integrate_pdf(d,e,4) - u_integrate_pdf(e,INFINITY,5);    
	}	
}

double u_path_14(double p,double a,double e,double d, double* exp, double* var, double *crossprob)
{
	*exp = 0;
	*var = 0;
	*crossprob = 0;
	
	// vertex order AE (only one possible)  
	*exp = u_expected_value(0,a,6) - u_expected_value(a,e,3);
	*var = u_second_moment(0,a,6) - u_second_moment(a,e,3);
	*crossprob = u_integrate_pdf(0,a,6) - u_integrate_pdf(a,e,3);	
}

double u_path_15(double p,double a,double e,double d, double* exp, double* var, double *crossprob)
{

	*exp = 0;
	*var = 0;
	*crossprob = 0;
	
 	// vertex order DP (only one possible)
	*exp = -u_expected_value(0,d,6) - u_expected_value(d,p,1);	
	*var = -u_second_moment(0,d,6) - u_second_moment(d,p,1);	
	*crossprob = -u_integrate_pdf(0,d,6) - u_integrate_pdf(d,p,1);	
	
}

double u_path_16(double p,double a,double e,double d, double* exp, double* var, double *crossprob)
{
	*exp = 0;
	*var = 0;
	*crossprob = 0;
	
	// Vertex order DA
	if (u_le(d,a))
	{   
	*exp = u_expected_value(0,d,7) + u_expected_value(0,d,8) + u_expected_value(d,a,7) + u_expected_value(d,a,10) + u_expected_value(a,INFINITY,9) + u_expected_value(a,INFINITY,10);
	*var = u_second_moment(0,d,7) + u_second_moment(0,d,8) + u_second_moment(d,a,7) + u_second_moment(d,a,10) + u_second_moment(a,INFINITY,9) + u_second_moment(a,INFINITY,10);
	*crossprob = u_integrate_pdf(0,d,7) + u_integrate_pdf(0,d,8) + u_integrate_pdf(d,a,7) + u_integrate_pdf(d,a,10) + u_integrate_pdf(a,INFINITY,9) + u_integrate_pdf(a,INFINITY,10);  
	}
  
	// Vertex order AD
	else if (u_le(a,d))
	{     
	*exp = u_expected_value(0,a,7) + u_expected_value(0,a,8) + u_expected_value(a,d,9) + u_expected_value(a,d,8) + u_expected_value(d,INFINITY,9) + u_expected_value(d,INFINITY,10);
	*var = u_second_moment(0,a,7) + u_second_moment(0,a,8) + u_second_moment(a,d,9) + u_second_moment(a,d,8) + u_second_moment(d,INFINITY,9) + u_second_moment(d,INFINITY,10);
	*crossprob = u_integrate_pdf(0,a,7) + u_integrate_pdf(0,a,8) + u_integrate_pdf(a,d,9) + u_integrate_pdf(a,d,8) + u_integrate_pdf(d,INFINITY,9) + u_integrate_pdf(d,INFINITY,10);    
	}	
}

double u_path_17(double p,double a,double e,double d, double* exp, double* var, double *crossprob)
{

	*exp = 0;
	*var = 0;
	*crossprob = 0;
	
	*exp = u_expected_value(a,INFINITY,2);	
	*var = u_second_moment(a,INFINITY,2);	
	*crossprob = u_integrate_pdf(a,INFINITY,2);	
	
}

double u_path_18(double p,double a,double e,double d, double* exp, double* var, double *crossprob)
{

	*exp = 0;
	*var = 0;
	*crossprob = 0;
	
	*exp = u_expected_value(d,INFINITY,4);	
	*var = u_second_moment(d,INFINITY,4);	
	*crossprob = u_integrate_pdf(d,INFINITY,4);	
	
}

double u_path_19(double p,double a,double e,double d, double* exp, double* var, double *crossprob)
{

	*exp = 0;
	*var = 0;
	*crossprob = 0;

	*exp = u_expected_value(a,INFINITY,2);	
	*var = u_second_moment(a,INFINITY,2);	
	*crossprob = u_integrate_pdf(a,INFINITY,2);		

}

double u_path_20(double p,double a,double e,double d, double* exp, double* var, double *crossprob)
{

	*exp = 0;
	*var = 0;
	*crossprob = 0;
	
	*exp = u_expected_value(a,e,2) - u_expected_value(e,INFINITY,6);	
	*var = u_second_moment(a,e,2) - u_second_moment(e,INFINITY,6);	
	*crossprob = u_integrate_pdf(a,e,2) - u_integrate_pdf(e,INFINITY,6);		
}

double u_path_21(double p,double a,double e,double d, double* exp, double* var, double *crossprob)
{

	*exp = 0;
	*var = 0;
	*crossprob = 0;

	// vertex order DP
	*exp = u_expected_value(0,d,3) - u_expected_value(d,p,1);
	*var = u_second_moment(0,d,3) - u_second_moment(d,p,1);
	*crossprob = u_integrate_pdf(0,d,3) - u_integrate_pdf(d,p,1);	
}

double u_path_22(double p,double a,double e,double d, double* exp, double* var, double *crossprob)
{

	*exp = 0;
	*var = 0;
	*crossprob = 0;

	*exp = u_expected_value(d,INFINITY,4);	
	*var = u_second_moment(d,INFINITY,4);	
	*crossprob = u_integrate_pdf(d,INFINITY,4);		
}

double u_path_23(double p,double a,double e,double d, double* exp, double* var, double *crossprob)
{
	*exp = 0;
	*var = 0;
	*crossprob = 0;
	
	// vertex order DP

	*exp = u_expected_value(d,p,4) + u_expected_value(p,INFINITY,6);
	*var = u_second_moment(d,p,4) + u_second_moment(p,INFINITY,6);
	*crossprob = u_integrate_pdf(d,p,4) + u_integrate_pdf(p,INFINITY,6);	
}

double u_path_24(double p,double a,double e,double d, double* exp, double* var, double *crossprob)
{

	*exp = 0;
	*var = 0;
	*crossprob = 0;
	
	// vertex order AE
	*exp = u_expected_value(0,a,1) - u_expected_value(a,e,3);
	*var = u_second_moment(0,a,1) - u_second_moment(a,e,3);
	*crossprob = u_integrate_pdf(0,a,1) - u_integrate_pdf(a,e,3);	
}


void u_pdf_piece(double p,double a,double e, double d, double* exp, double* var, double* crossprob)
{
	double expected_pdf_piece = 0;

// Provide the coordinates of the the  the vertices, and coordinates of
// leftmost and the rightmost intersection of the parallelogram with 
//the horizontal axis.

double Px, Py, Ax, Ay, Ex, Ey, Dx, Dy, Z2right, Z2left;

// PQRS

// Following is the mapping

// P -> P
// Q -> A
// R -> E
// S -> D
	Px = u_mu2-u_mu1+u_delta2-u_delta1;
	Py = u_c-u_mu1-u_delta1;
	Ax = u_mu2-u_mu1+u_delta1+u_delta2;
	Ay = u_c-u_mu1+u_delta1;
	Ex = u_mu2-u_mu1+u_delta1-u_delta2;
	Ey = u_c-u_mu1+u_delta1;
	Dx = u_mu2-u_mu1-u_delta1-u_delta2;
	Dy = u_c-u_mu1-u_delta1;
	// intersection of AP with with horizontal axis (Z2). 
	Z2right = u_mu2-u_c+u_delta2;
	// intersection of ED with with horizontal axis (Z2). 
	Z2left = u_mu2-u_c-u_delta2;




//        e -  -  - a
//        /         /
//       /   P1    /
//      /         /
//     d- - - - - p

//% P1 is a polynomial.


//-------------------------------------------------------------------------
 // sets of 4 positive vertices 
 
// Slopes of P, A, E, D all are greater than 0. There are 5 possible parallelogram configurations for
// when all slopes are are greater than 0.

//  If whole parallelogram lies in the first quadrant   
if (u_ge(p,0) && u_ge(Px,0) && u_ge(Py,0) && u_ge(a,0) && u_ge(Ax,0) && u_ge(Ay,0) && u_ge(e,0) && u_ge(Ex,0) && u_ge(Ey,0) && u_ge(d,0) && u_ge(Dx,0) && u_ge(Dy,0))

    u_path_1(p,a,e,d,exp,var,crossprob);

//  If whole parallelogram lies in the third quadrant
else if (u_ge(p,0) && u_le(Px,0) && u_le(Py,0) && u_ge(a,0) && u_le(Ax,0) && u_le(Ay,0) && u_ge(e,0) && u_le(Ex,0) && u_le(Ey,0) && u_ge(d,0) && u_le(Dx,0) && u_le(Dy,0))
    
    u_path_2(p,a,e,d,exp,var,crossprob);
    
//  If E,A lie in the first quadrant and D,P lie in the third quadrant assuming edge AP crosses second quadrant;
else if (u_ge(p,0) && u_le(Px,0)  && u_le(Py,0) && u_ge(a,0) && u_ge(Ax,0) && u_ge(Ay,0) && u_ge(e,0) && u_ge(Ex,0) && u_ge(Ey,0) && u_ge(d,0) && u_le(Dx,0) && u_le(Dy,0) && u_le(Z2right,0))
    
    u_path_3(p,a,e,d,exp,var,crossprob);
    
//  If E,A lie in the first quadrant and D,P lie in the third quadrant assuming edge ED crosses fourh quadrant;
else if (u_ge(p,0) && u_le(Px,0) && u_le(Py,0) && u_ge(a,0) && u_ge(Ax,0) && u_ge(Ay,0) && u_ge(e,0) && u_ge(Ex,0) && u_ge(Ey,0) && u_ge(d,0) && u_le(Dx,0) && u_le(Dy,0) && u_ge(Z2left,0))

    u_path_4(p,a,e,d,exp,var,crossprob);	
   
//  If E,A lie in the first quadrant and D,P lie in the third quadrant assuming edge ED crosses second quadrant
//  and AP crosses the fourth quadrant;
else if (u_ge(p,0) && u_le(Px,0) && u_le(Py,0) && u_ge(a,0) && u_ge(Ax,0) && u_ge(Ay,0) && u_ge(e,0) && u_ge(Ex,0) && u_ge(Ey,0) && u_ge(d,0) && u_le(Dx,0) && u_le(Dy,0) && u_le(Z2left,0) && u_ge(Z2right,0))

    u_path_5(p,a,e,d,exp,var,crossprob);
    
//------------------------------------------------------------------------
    // sets of 3 positive vertices 
    
    // Slopes of P, A, D are greater than 0 and slope of E is less than 0. There are 2 possible parallelogram configurations
    // in this case.
    
    // If D, P lie in the third quadrant,A in first and E in second assuming edge ED
    // and AP cross the second quadrant;
else if (u_ge(p,0) && u_le(Px,0) && u_le(Py,0) && u_ge(a,0) && u_ge(Ax,0) && u_ge(Ay,0) && u_le(e,0) && u_le(Ex,0) && u_ge(Ey,0) && u_ge(d,0) && u_le(Dx,0)  && u_le(Dy,0) && u_le(Z2right,0))
    
    u_path_6(p,a,e,d,exp,var,crossprob);
    
    // If D, P lie in the third quadrant,A in first and E in second assuming edge ED crosses second quadrant
    // and AP crosses the fourth quadrant;
else if (u_ge(p,0) && u_le(Px,0) && u_le(Py,0) && u_ge(a,0) && u_ge(Ax,0) && u_ge(Ay,0) && u_le(e,0) && u_le(Ex,0) && u_ge(Ey,0) && u_ge(d,0) && u_le(Dx,0) && u_le(Dy,0) && u_le(Z2left,0) && u_ge(Z2right,0))
    
    u_path_7(p,a,e,d,exp,var,crossprob);
    
    // Slopes of A, E, D are greater than 0 and slope of P is less than 0. There are 2 possible parallelogram configurations
    //in this case.
    
    // If E, A lie in the first quadrant,D in third and P in fourth assuming edge ED
    //and AP cross the fourth quadrant;
else if (u_le(p,0) && u_ge(Px,0) && u_le(Py,0) && u_ge(a,0) && u_ge(Ax,0) && u_ge(Ay,0) && u_ge(e,0) && u_ge(Ex,0) && u_ge(Ey,0) && u_ge(d,0) && u_le(Dx,0) && u_le(Dy,0) && u_ge(Z2left,0))
    
    u_path_8(p,a,e,d,exp,var,crossprob);
    
    // If E, A lie in the first quadrant,D in third and P in fourth assuming edge ED crosses second quadrant
    //and AP crosses the fourth quadrant;
else if (u_le(p,0) && u_ge(Px,0) && u_le(Py,0) && u_ge(a,0) && u_ge(Ax,0) && u_ge(Ay,0) && u_ge(e,0) && u_ge(Ex,0) && u_ge(Ey,0) && u_ge(d,0) && u_le(Dx,0)  && u_le(Dy,0) && u_le(Z2left,0) && u_ge(Z2right,0))
    
    u_path_9(p,a,e,d,exp,var,crossprob);
    
    // Newly added after debugging
    // Slopes of P, A, E are greater than 0 and slope of D is less than 0.
    // P, A, E in the first quadrant and D in the second quadrant 
else if (u_ge(p,0) && u_ge(Px,0) && u_ge(Py,0) && u_ge(a,0) && u_ge(Ax,0) && u_ge(Ay,0) && u_ge(e,0) && u_ge(Ex,0) && u_ge(Ey,0) && u_le(d,0) && u_le(Dx,0) && u_ge(Dy,0))
    
    u_path_10(p,a,e,d,exp,var,crossprob);
    
    // Slopes of E, D, P are greater than 0 and slope of A is less than 0.
    // E, D, P in the third quadrant and A in the fourth quadrant     
else if (u_ge(p,0) && u_le(Px,0) && u_le(Py,0) && u_le(a,0) && u_ge(Ax,0) && u_le(Ay,0) && u_ge(e,0) && u_le(Ex,0) && u_le(Ey,0) && u_ge(d,0) && u_le(Dx,0) && u_le(Dy,0))
    
    u_path_11(p,a,e,d,exp,var,crossprob);
    
   //------------------------------------------------------------------------
    // sets of 2 positive vertices 
    // There are 5 possible parallelogram configurations in this case.
    
// E,D in second quadrant and A,P in the first quadrant    
else if (u_ge(p,0) && u_ge(Px,0) && u_ge(Py,0) && u_ge(a,0) && u_ge(Ax,0) && u_ge(Ay,0) && u_le(e,0) && u_le(Ex,0)  && u_ge(Ey,0) && u_le(d,0) && u_le(Dx,0)  && u_ge(Dy,0))
   
    u_path_12(p,a,e,d,exp,var,crossprob);
    
// E,D in third quadrant and A,P in the fourth quadrant        
else if (u_le(p,0) && u_ge(Px,0) && u_le(Py,0) && u_le(a,0) && u_ge(Ax,0) && u_le(Ay,0) && u_ge(e,0) && u_le(Ex,0) && u_le(Ey,0) && u_ge(d,0) && u_le(Dx,0) && u_le(Dy,0))
   
    u_path_13(p,a,e,d,exp,var,crossprob);
    
// E,A in first quadrant and D,P in the fourth quadrant        
else if (u_le(p,0) && u_ge(Px,0) && u_le(Py,0) && u_ge(a,0) && u_ge(Ax,0) && u_ge(Ay,0) && u_ge(e,0) && u_ge(Ex,0)  && u_ge(Ey,0) && u_le(d,0) && u_ge(Dx,0) && u_le(Dy,0))
   
   u_path_14(p,a,e,d,exp,var,crossprob);
    
// E,A in second quadrant and D,P in the third quadrant           
else if (u_ge(p,0) && u_le(Px,0) && u_le(Py,0) && u_le(a,0) && u_le(Ax,0) && u_ge(Ay,0) && u_le(e,0) && u_le(Ex,0) && u_ge(Ey,0) && u_ge(d,0) && u_le(Dx,0) && u_le(Dy,0))
   
    u_path_15(p,a,e,d,exp,var,crossprob);
    
// E in second quadrant, P in fourth quadrant, A in the first quadrant and D in the third quadrant           
else if (u_le(p,0) && u_ge(Px,0) && u_le(Py,0) && u_ge(a,0) && u_ge(Ax,0) && u_ge(Ay,0) && u_le(e,0) && u_le(Ex,0)  && u_ge(Ey,0) && u_ge(d,0) && u_le(Dx,0) && u_le(Dy,0))
   
    u_path_16(p,a,e,d,exp,var,crossprob);
   
  //------------------------------------------------------------------------
    //  1 positive vertex possibilities 
    // There are 2 possible parallelogram configurations in this case.

// A in first quadrant, E,D,P in the second quadrant           
else if (u_le(p,0) && u_le(Px,0) && u_ge(Py,0) && u_ge(a,0) && u_ge(Ax,0) && u_ge(Ay,0) && u_le(e,0) && u_le(Ex,0) && u_ge(Ey,0) && u_le(d,0) && u_le(Dx,0)  && u_ge(Dy,0))
    
    u_path_17(p,a,e,d,exp,var,crossprob);
    
// D in third quadrant, E,A,P in the fourth quadrant           
else if (u_le(p,0) && u_ge(Px,0) && u_le(Py,0) && u_le(a,0) && u_ge(Ax,0) && u_le(Ay,0) && u_le(e,0) && u_ge(Ex,0) && u_le(Ey,0) && u_ge(d,0) && u_le(Dx,0)  && u_le(Dy,0))

    u_path_18(p,a,e,d,exp,var,crossprob);
    
    //------------------------------------------------------------------------
    //  special case where one of the vertex is at origin(0,0) i.e. when
    //  slope of one vertex is NAN
    //  There are 6 possible parallelogram configurations in this case.

    // P at origin, A in first quadrant, E in the second quadrant 
else if (u_eq(Px,0) && u_eq(Py,0) && u_ge(a,0) && u_ge(Ax,0) && u_ge(Ay,0) && u_le(e,0) && u_le(Ex,0) && u_ge(Ey,0))
    
     u_path_19(p,a,e,d,exp,var,crossprob);
    
   // P at origin, E,A in first quadrant
else if (u_eq(Px,0) && u_eq(Py,0) && u_ge(a,0) && u_ge(Ax,0) && u_ge(Ay,0) && u_ge(e,0) && u_ge(Ex,0) && u_ge(Ey,0)) 
    
     u_path_20(p,a,e,d,exp,var,crossprob);
    
   // A at origin, D,P in the third quadrant 
else if (u_ge(p,0) && u_le(Px,0) && u_le(Py,0) && u_eq(Ax,0) && u_eq(Ay,0) && u_ge(d,0) && u_le(Dx,0) && u_le(Dy,0))
    
     u_path_21(p,a,e,d,exp,var,crossprob);
    
   // E at origin, D in the third quadrant and P in the fourth quadrant 
else if (u_le(p,0) && u_ge(Px,0) && u_le(Py,0) && u_eq(Ex,0) && u_eq(Ey,0) && u_ge(d,0) && u_le(Dx,0) && u_le(Dy,0))
    
     u_path_22(p,a,e,d,exp,var,crossprob);
    
   // E at origin, D,P in the third quadrant 
else if (u_ge(p,0) && u_le(Px,0) && u_le(Py,0) && u_eq(Ex,0) && u_eq(Ey,0) && u_ge(d,0) && u_le(Dx,0) && u_le(Dy,0)) 
    
     u_path_23(p,a,e,d,exp,var,crossprob);
    
   // D at origin, E,A in the first quadrant  
else if (u_ge(a,0) && u_ge(Ax,0) && u_ge(Ay,0) && u_ge(e,0) && u_ge(Ex,0) && u_ge(Ey,0) && u_eq(Dx,0) && u_eq(Dy,0))  

     u_path_24(p,a,e,d,exp,var,crossprob);

// Added during debugging
//If one of the e_paths is not taken means subparallelogram lies in the second/fourth quadrant where all slope values
// are negative		
else
{
	// no contribution to expected value, variance and crossing probability from the subparallelogram
	*exp = 0;
	*var = 0;
	*crossprob = 0;
}     	

}

double u_determine_slope_value(double z1, double z2)
{

	double slope_temp;

	if(fabs(z2) < DBL_EPSILON)
	{
		// Assign NAN value if both z1 and z2 are 0
		if(fabs(z1) < DBL_EPSILON)
		{
			slope_temp = std::numeric_limits<double>::quiet_NaN();
		}
		else if (z1 > 0)
		{
			slope_temp = INFINITY;
		}
		else
		{
			slope_temp = -INFINITY;
		}
	}
	else
	{
		slope_temp = (double)((z1) / (z2));
	}
	
	return slope_temp;

}


double z_density_uniform :: alpha_density_uniform(double m1, double d1, double  m2, double  d2, double isovalue, double*  exp, double*  var, double*  crossprob)
{

	u_mu1 = m1;
	u_delta1 = d1;
	u_mu2 = m2;
	u_delta2 = d2;
	u_c = isovalue;
	
	double slope_OP, slope_OQ, slope_OR, slope_OS, slope_OA, slope_OB, slope_OC, slope_OD, slope_OE, mean_temp, delta_temp;

	// Assume e_mu2 is always greater than e_mu1
	if (u_mu1 > u_mu2)
	{
		
		mean_temp = u_mu1;
		u_mu1 = u_mu2;
		u_mu2 = mean_temp;

		delta_temp = u_delta1;
		u_delta1 = u_delta2;
		u_delta2 = delta_temp;
	}   

	slope_OP = u_determine_slope_value(u_c - u_mu1 - u_delta1, u_mu2 - u_mu1 + u_delta2 - u_delta1);
	slope_OQ = u_determine_slope_value(u_c - u_mu1 + u_delta1, u_mu2 - u_mu1 + u_delta2 + u_delta1);
	slope_OS = u_determine_slope_value(u_c - u_mu1 - u_delta1, u_mu2 - u_mu1 - u_delta2 - u_delta1);
	slope_OR = u_determine_slope_value(u_c - u_mu1 + u_delta1, u_mu2 - u_mu1 + u_delta1 - u_delta2);
	

	double expected = 0;
	double crossP = 0;
	double secondM = 0; 
	

	// set parameters for computing the polynomials. Use follwing global object in whole program.
	//tri_poly = new triangular_kernel_polynomial(e_mu1,e_delta1,e_mu2,e_delta2,c);
	uni_poly = new uniform_kernel_polynomial(u_mu1,u_delta1,u_mu2,u_delta2,u_c);


	// Parallelogram PQRS (only one polynomial over whole parallelogram)
        u_pdf_piece(slope_OP, slope_OQ, slope_OR, slope_OS, exp, var, crossprob);
	expected += *exp; 
	crossP += *crossprob;
	secondM += *var;   

	return crossP;
}

// set number of uniform distributions in kde1 
void z_density_uniform :: setNumUniformInKde1(int a)
{
	numUniformInKde1 = a;
}

// get number of uniform distributions in kde2 
int z_density_uniform :: getNumUniformInKde1()
{
	return numUniformInKde1;
}

// set number of uniform distributions in kde1 
void z_density_uniform :: setNumUniformInKde2(int a)
{
	numUniformInKde2 = a;
}

// get number of uniform distributions in kde2 
int z_density_uniform :: getNumUniformInKde2()
{
	return numUniformInKde2;
}

double z_density_uniform :: kde_z_pdf_expected(float* mu1, double h1, float* mu2, double h2, double c)
{
	int numKde1 = getNumUniformInKde1();
	int numKde2 = getNumUniformInKde2();

	double expected, c_prob, sig, second, first;

	double kde_expected = 0;

	double total_crossing_prob = 0;
	
	double rangemin,rangemax;
	
	//double EPSILON = 0.000000001;

	for(int i=0; i<numKde1; i++)
	{
		for(int j=0; j<numKde2; j++)
		{
			
			// If c is in the total range then only call alpha_density_triangular (Latest change)
			if((mu1[i]-h1) < (mu2[j]-h2))
				rangemin = (mu1[i]-h1);
			else
				rangemin = (mu2[j]-h2);
	
			if((mu1[i]+h1) > (mu2[j]+h2))
				rangemax = (mu1[i]+h1);
			else
				rangemax = (mu2[j]+h2);
	
	 		if((c < rangemin) || (c > rangemax))
		 	{
		 		expected = 0;
		 		c_prob =0; 
		 		sig = 0;		 	
			}
			else
		  	{
				alpha_density_uniform(mu1[i], h1, mu2[j], h2, c,&expected,&sig,&c_prob);
			}

			// What happens when crossing probability is 0 skip
			if(mu1[i] > mu2[j])
			{
				if(fabs(c_prob)<DBL_EPSILON)
				{
					// Means given pair of kernels doesn't contribute to the expected value
				}
				else
				{
					//kde_expected = kde_expected + (1 - expected); If pdf is not normalized 1 should be replace with crossing probability
					// Major update
					kde_expected = kde_expected + (c_prob - expected);
					total_crossing_prob = total_crossing_prob + c_prob;
				}
			}	
			else
			{
				if(fabs(c_prob)<DBL_EPSILON)
				{
					// Means given pair of kernels doesn't contribute to the expected value
				}
				else
				{
					kde_expected =  kde_expected + expected;
					total_crossing_prob = total_crossing_prob + c_prob;
				}
			}
			
		}
	}

  if(total_crossing_prob > 0)
  
		kde_expected = (double)(kde_expected/total_crossing_prob);

  else
		
		kde_expected = 0.5;
		

  return kde_expected;
}



// The weights for each of the kernels were obtaines using the nonlocal means technique. Triangular kernel is used for the nonparametric density estimation.
double z_density_uniform :: kde_z_pdf_expected_NL(float* mu1, float* mu1_wts, double h1, float* mu2, float* mu2_wts, double h2, double c)
{
	int numKde1 = getNumUniformInKde1();
	int numKde2 = getNumUniformInKde2();

	double expected, c_prob, sig, second, first;

	double kde_expected = 0;

	double total_crossing_prob = 0;
	
	double rangemin,rangemax;
	
	//double EPSILON = 0.000000001;

	for(int i=0; i<numKde1; i++)
	{
		for(int j=0; j<numKde2; j++)
		{
			//temp = alpha_pdf(mu1[i], h1, mu2[j], h2, c);
	
			//Compute0To1(temp,&expected,&c_prob,&sig,&second,&first);

			//cout<<"c_prob:"<<c_prob<<"\n";
			
			// If c is in the total range then only call alpha_density_triangular (Latest change)
			if((mu1[i]-h1) < (mu2[j]-h2))
				rangemin = (mu1[i]-h1);
			else
				rangemin = (mu2[j]-h2);
	
			if((mu1[i]+h1) > (mu2[j]+h2))
				rangemax = (mu1[i]+h1);
			else
				rangemax = (mu2[j]+h2);
	
	 		if((c < rangemin) || (c > rangemax))
		 	{
		 		expected = 0;
		 		c_prob =0; 
		 		sig = 0;		 	
			}
			else
		  {
				alpha_density_uniform(mu1[i], h1, mu2[j], h2, c,&expected,&sig,&c_prob);
				
				// the pdf gets scaled by the product of the weights of the kernels. Therefore, all the quantities
				// expected (first moment), sig(second moment), and c_prob get scaled by that factor. (We assume that weights for the kernels in pdf are normalized.)
				expected  = expected*mu1_wts[i]*mu2_wts[j];
				sig  = sig*mu1_wts[i]*mu2_wts[j];
				c_prob  = c_prob*mu1_wts[i]*mu2_wts[j];

			}

			// What happens when crossing probability is 0 skip
			if(mu1[i] > mu2[j])
			{
				if(fabs(c_prob)<DBL_EPSILON)
				{
					// Means given pair of kernels doesn't contribute to the expected value
				}
				else
				{
					//kde_expected = kde_expected + (1 - expected); If pdf is not normalized 1 should be replace with crossing probability
					// Major update
					kde_expected = kde_expected + (c_prob - expected);
					total_crossing_prob = total_crossing_prob + c_prob;
				}
			}	
			else
			{
				if(fabs(c_prob)<DBL_EPSILON)
				{
					// Means given pair of kernels doesn't contribute to the expected value
				}
				else
				{
					kde_expected =  kde_expected + expected;
					total_crossing_prob = total_crossing_prob + c_prob;
				}
			}
			
		}
	}

  if(total_crossing_prob > 0)
  
		kde_expected = (double)(kde_expected/total_crossing_prob);

	else
		
		kde_expected = 0.5;
	
	return kde_expected;
}


double z_density_uniform :: kde_z_pdf_variance(float* mu1, double h1, float* mu2, double h2, double c, double expected_crossing)
{

	int numKde1 = getNumUniformInKde1();
	int numKde2 = getNumUniformInKde2();
	double expected, c_prob, second;

	double kde_e_second_moment = 0;
	double total_crossing_prob = 0;	

	//double kde_first_moment = kde_z_pdf_expected(mu1, h1, mu2, h2, c);
	double kde_first_moment = expected_crossing;	
	
	double kde_variance = 0;
	
	double rangemin,rangemax;

	for(int i=0; i<numKde1; i++)
	{
		for(int j=0; j<numKde2; j++)
		{
			// If c is in the total range then only call alpha_density_triangular (Latest change)
			if((mu1[i]-h1) < (mu2[j]-h2))
				rangemin = (mu1[i]-h1);
			else
				rangemin = (mu2[j]-h2);
	
			if((mu1[i]+h1) > (mu2[j]+h2))
				rangemax = (mu1[i]+h1);
			else
				rangemax = (mu2[j]+h2);
	
	 		if((c < rangemin) || (c > rangemax))
		 	{
		 		expected = 0;
		 		c_prob =0; 
		 		second = 0;		 	
			}
			else
		        {		
				alpha_density_uniform(mu1[i], h1, mu2[j], h2, c,&expected,&second,&c_prob);
			}

			if(mu1[i] > mu2[j])
			{
					if(fabs(c_prob)<DBL_EPSILON)
					{
						// kde_e_second_moment = kde_e_second_moment + 0.1;
						// Means given pair of kernels doesn't determine variance
					}
					else
					{
						//If pdf is not normalized 1 should be replace with crossing probability	
						kde_e_second_moment = kde_e_second_moment + second + c_prob - 2*expected;
						total_crossing_prob = total_crossing_prob + c_prob;
					}
			}	
			else
			{
					if(fabs(c_prob)<DBL_EPSILON)
					{
						// kde_e_second_moment = kde_e_second_moment + 0.1;
						// Means given pair of kernels doesn't determine variance
					}
					else
					{
						kde_e_second_moment =  kde_e_second_moment + second;
						total_crossing_prob = total_crossing_prob + c_prob;
					}
			}
			
		}
	}

	if(total_crossing_prob > 0)
	
		kde_variance  = (double)(kde_e_second_moment/total_crossing_prob) - kde_first_moment*kde_first_moment;

	else
		
		kde_variance = 0;

	return kde_variance;
}


// The weights for each of the kernels were obtaines using the nonlocal means technique. Triangular kernel is used for the nonparametric density estimation.
double z_density_uniform :: kde_z_pdf_variance_NL(float* mu1, float* mu1_wts, double h1, float* mu2, float* mu2_wts, double h2, double c, double expected_crossing)
{

	int numKde1 = getNumUniformInKde1();
	int numKde2 = getNumUniformInKde2();
	double expected, c_prob, second;

	double kde_second_moment = 0;
	double total_crossing_prob = 0;	

	//double kde_first_moment = kde_z_pdf_expected(mu1, h1, mu2, h2, c);
	double kde_first_moment = expected_crossing;	
	
	double kde_variance = 0;
	
	double rangemin,rangemax;

	for(int i=0; i<numKde1; i++)
	{
		for(int j=0; j<numKde2; j++)
		{
			
			// If c is in the total range then only call alpha_density_triangular (Latest change)
			if((mu1[i]-h1) < (mu2[j]-h2))
				rangemin = (mu1[i]-h1);
			else
				rangemin = (mu2[j]-h2);
	
			if((mu1[i]+h1) > (mu2[j]+h2))
				rangemax = (mu1[i]+h1);
			else
				rangemax = (mu2[j]+h2);
	
	 		if((c < rangemin) || (c > rangemax))
		 	{
		 		expected = 0;
		 		c_prob =0; 
		 		second = 0;		 	
			}
			else
		  {		
				alpha_density_uniform(mu1[i], h1, mu2[j], h2, c,&expected,&second,&c_prob);
				// the pdf gets scaled by the product of the weights of the kernels. Therefore, all the quantities
				// expected (first moment), sig(second moment), and c_prob get scaled by that factor. (We assume that weights for the kernels in pdf are normalized.)
				expected  = expected*mu1_wts[i]*mu2_wts[j];
				second  = second*mu1_wts[i]*mu2_wts[j];
				c_prob  = c_prob*mu1_wts[i]*mu2_wts[j];

			}

			if(mu1[i] > mu2[j])
			{
					if(fabs(c_prob)<DBL_EPSILON)
					{
						// kde_second_moment = kde_second_moment + 0.1;
						// Means given pair of kernels doesn't determine variance
					}
					else
					{
						//If pdf is not normalized 1 should be replace with crossing probability	
						kde_second_moment = kde_second_moment + second + c_prob - 2*expected;
						total_crossing_prob = total_crossing_prob + c_prob;
					}
			}	
			else
			{
					if(fabs(c_prob)<DBL_EPSILON)
					{
						// kde_second_moment = kde_second_moment + 0.1;
						// Means given pair of kernels doesn't determine variance
					}
					else
					{
						kde_second_moment =  kde_second_moment + second;
						total_crossing_prob = total_crossing_prob + c_prob;
					}
			}
			
		}
	}

	if(total_crossing_prob > 0)
	
		kde_variance  = (double)(kde_second_moment/total_crossing_prob) - kde_first_moment*kde_first_moment;

	else
		
		kde_variance = 0;

	return kde_variance;
}
