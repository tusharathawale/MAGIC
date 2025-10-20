#ifndef EASY_LIALG_EQUATION
#define EASY_LIALG_EQUATION

#include "./basic.h"
#include "./blas.h"

// This header files contains function for solving Ax=b
namespace EASYLINALG
{
    // refer to
    // https://www.cs.upc.edu/~jordicf/Teaching/programming/pdf4/MATH03_Gaussian-4slides.pdf
    template <typename T, uint Size>
    LIAG_FUNC_MACRO Vec<T, Size> SymmBackSubstitution(const Matrix<T, Size, Size> &A, const Vec<T, Size> &b)
    {
        Vec<double, Size> x(0.0);
        // TODO, double check this, in beetle beetle_124_208_208 some value is 0.0001848
        bool ifUpper = A.IsUpperTriangular(0.005);
        // TODO, print sth if the A is not upper triangular
        //if(ifUpper == false){
        //    A.Show();
        //}
        assert(ifUpper == true);

        for (int i = Size - 1; i >= 0; --i)
        {
            // be careful about the last iteration here
            // it might hit the case where i =-1
            // in that case, there is error
            if (i < 0)
            {
                break;
            }
            double s = 0;
            // start with one element after i
            // end with the last element

            for (uint j = (i + 1); j < Size; ++j)
            {
                s = s + A[i][j] * x[j];
            }

            // at the last iteration, i is goes to -1
            x[i] = (b[i] - s) / A[i][i];
        }
        return x;
    }

}

#endif