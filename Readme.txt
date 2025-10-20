/*============================================================================

Title   -  MAGIC: Marching Cubes Isosurface Uncertainty Visualization for Gaussian Uncertain Data with Spatial Correlation
Authors -  Tushar M. Athawale, Kenneth Moreland, David Pugmire, Chris R. Johnson, Paul Rosen, Matthew Norman, Antigoni Georgiadou, and Alireza Entezari
Date    -  Oct 20, 2025

This code is distributed under the OSI-approved BSD 3-clause License. See LICENSE.txt for details.
 
============================================================================*/

- Navigate to the "independent_Gaussian" directory for isosurface extraction using the independent Gaussian model
[The MAGIC paper] 

- Navigate to the "correlated_Gaussian" directory for isosurface extraction using the Gaussian model with spatial correlation
[The MAGIC paper] 

- Navigate to the "kde_independent_Gaussian" directory for isosurface extraction using the kernel density estimation with independent Gaussian kernel
[The MAGIC paper]  

- Navigate to the "kde_correlated_Gaussian" directory for isosurface extraction using the kernel density estimation with correlated Gaussian kernel
[The MAGIC paper] 

- Navigate to the "independent_uniform" directory for isosurface extraction using the uniform distribution model

Reference:
 T. Athawale and A. Entezari, "Uncertainty Quantification in Linear Interpolation for Isosurface Extraction,"
 in IEEE Transactions on Visualization and Computer Graphics, vol. 19, no. 12, pp. 2723-2732, Dec. 2013,
 doi: 10.1109/TVCG.2013.208.

- Navigate to the "kde_independent_uniform" directory for isosurface extraction using the kernel density estimation with independent uniform kernel

Reference:
 T. Athawale, E. Sakhaee and A. Entezari, "Isosurface Visualization of Data with Nonparametric Models for 
 Uncertainty," in IEEE Transactions on Visualization and Computer Graphics, vol. 22, no. 1, pp. 777-786, 
 31 Jan. 2016, doi: 10.1109/TVCG.2015.2467958.

- Navigate to the "crossing_probability_entropy" directory for computation of cell-crossing probability and entropy.

References:
1) Z. Wang, T. M. Athawale, K. Moreland, J. Chen, C. R. Johnson, and
 D. Pugmire, “FunMC^2: A Filter for Uncertainty Visualization of
 Marching Cubes on Multi-Core Devices,” in Eurographics Symposium
 on Parallel Graphics and Visualization, R. Bujack, D. Pugmire, and
 G. Reina, Eds. The Eurographics Association, 2023.
 doi: 10.2312/pgv.20231081
 
(2) T. M. Athawale, S. Sane and C. R. Johnson, "Uncertainty Visualization
 of the Marching Squares and Marching Cubes Topology Cases," in 2021 IEEE
 Visualization Conference (VIS), New Orleans, LA, USA, 2021, pp. 106-110,
 doi: 10.1109/VIS49827.2021.9623267.
 
(3) K. P¨othkow, B. Weber, and H.-C. Hege, “Probabilistic marching cubes,”
 Computer Graphics Forum, vol. 30, no. 3, pp. 931–940, 2011.
 doi: 10.1111/j.1467-8659.2011.01942.x

- Navigate to the "eigen_based_crossing_probability_entropy" directory for computation of eigenvalue-based (data-driven) cell-crossing probability and entropy.

Reference:
 T. M. Athawale, Z. Wang, C. R. Johnson, and D. Pugmire, “Data-Driven
 Computation of Probabilistic Marching Cubes for Efficient Visualization
 of Level-Set Uncertainty,” in EuroVis (Short Papers), C. Tominski,
 M. Waldner, and B. Wang, Eds. The Eurographics Association, 2024.
 doi: 10.2312/evs.20241071

Note: The synthetic tangle dataset is provided in each subfolder for running the code and creating sample results. The following paper has details relevant to creating the tangle dataset.

 A. Knoll, Y. Hijazi, A. Kensler, M. Schott, C. Hansen, and H. Hagen,
 “Fast ray tracing of arbitrary implicit surfaces with interval and affine
 arithmetic,” Computer Graphics Forum, vol. 28, no. 1, pp. 26–40, 2009.