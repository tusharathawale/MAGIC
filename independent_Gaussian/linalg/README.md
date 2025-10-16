
This folder contains all kinds of matrix operations

Related files:

https://rosettacode.org/wiki/QR_decomposition#C

https://github.com/madrury/linalg/blob/master/eigen.c

https://rpubs.com/aaronsc32/qr-decomposition-householder 
(it seems that the sign can be eigher positive or negative)

https://www.cs.cornell.edu/~bindel/class/cs6210-f09/lec18.pdf


TODO, optimize the pipeline to aovid the memory issue

 malloc will assign memory in global space, 
 maybe update the code and try to do things in the stack space

 
1 check the qr code, and decrease the mem allocation
2 simplify the sampling operation, instead of malloc 1k size matrix at one time
  change it into the serial version

3 update the hole pipeline, copy back A matrix and the mean array to cpu then generating a long array, each process one sampling operation, then using the reduce by key to reduce the results and get the value per cell

4 divide the whole data set to several small data if the optimization still can not solve the memory issue